"""
Feature extraction: BPM detection, timing-grid filtering, density control,
silence filtering, column mapping, column balance, physical corrections, and
hold-note normalization.
"""

import numpy as np
from scipy import signal

from .config import Config


class FeatureExtractor:
    def __init__(self, config=None):
        self.config = config or Config()

    def detect_bpm(self, y, sr):
        print("检测BPM...")
        frame_length = self.config.N_FFT
        hop_length = self.config.HOP_LENGTH

        if len(y) < frame_length:
            bpm = 120.0
            beats = np.array([0.0])
            first_beat_time = 0.0
            print(f"检测到BPM: {bpm:.1f}, 第一个节拍: {first_beat_time:.2f}s")
            return {"bpm": bpm, "first_beat_time": first_beat_time, "beats": beats}

        frame_count = 1 + (len(y) - frame_length) // hop_length
        rms_values = np.empty(frame_count, dtype=np.float32)

        for index in range(frame_count):
            start = index * hop_length
            frame = y[start : start + frame_length]
            rms_values[index] = np.sqrt(np.mean(np.square(frame)))

        onset_envelope = np.maximum(0.0, np.diff(rms_values, prepend=rms_values[0]))
        onset_envelope -= np.mean(onset_envelope)
        onset_envelope = np.maximum(onset_envelope, 0.0)

        autocorr = np.correlate(onset_envelope, onset_envelope, mode="full")
        autocorr = autocorr[len(onset_envelope) - 1 :]

        min_bpm = 60
        max_bpm = 240
        lag_min = max(1, int(round((60 * sr) / (max_bpm * hop_length))))
        lag_max = max(lag_min + 1, int(round((60 * sr) / (min_bpm * hop_length))))
        search_window = autocorr[lag_min:lag_max]

        if search_window.size == 0 or np.allclose(search_window, 0):
            bpm = 120.0
        else:
            best_lag = lag_min + int(np.argmax(search_window))
            bpm = 60.0 * sr / (best_lag * hop_length)

        peak_distance = max(1, int(round((60 * sr) / (bpm * hop_length))))
        peak_indices, _ = signal.find_peaks(
            onset_envelope,
            distance=max(1, int(peak_distance * 0.8)),
            prominence=max(np.std(onset_envelope) * 0.5, 1e-6),
        )

        if peak_indices.size == 0:
            beats = np.array([0.0])
        else:
            beats = peak_indices * hop_length / sr

        first_beat_time = float(beats[0]) if beats.size > 0 else 0.0
        print(f"检测到BPM: {bpm:.1f}, 第一个节拍: {first_beat_time:.2f}s")
        return {"bpm": bpm, "first_beat_time": first_beat_time, "beats": beats}

    def _snap_time_to_grid(self, time_ms, beat_duration_ms, first_beat_ms, divisors):
        if beat_duration_ms <= 0 or not divisors:
            return float(time_ms), 0, 0.0

        raw_beat_pos = (float(time_ms) - float(first_beat_ms)) / float(beat_duration_ms)
        candidates = []

        for divisor in divisors:
            snapped_pos = round(raw_beat_pos * divisor) / divisor
            error_ms = abs(raw_beat_pos - snapped_pos) * beat_duration_ms
            candidates.append((float(error_ms), int(divisor), float(snapped_pos)))

        if not candidates:
            return float(time_ms), 0, 0.0

        min_error = min(candidate[0] for candidate in candidates)
        coarse_bias_ms = min(
            max(beat_duration_ms / 18.0, 6.0),
            max(float(self.config.MAX_ALIGN_ERROR_MS) * 0.22, 6.0),
        )
        preferred_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] <= min_error + coarse_bias_ms
        ]
        preferred_candidates.sort(key=lambda candidate: (candidate[1], candidate[0]))
        chosen_error, best_divisor, best_snap = preferred_candidates[0]

        snapped_time = float(first_beat_ms + best_snap * beat_duration_ms)
        return snapped_time, best_divisor, float(chosen_error)

    def align_to_beat_grid(self, note_events, bpm, first_beat_time=0):
        print("多级节拍网格对齐...")
        aligned_notes = []
        beat_duration_ms = 60000 / bpm if bpm else 0
        first_beat_ms = first_beat_time * 1000
        valid_count = 0

        for source_note in note_events:
            note = dict(source_note)
            start_time = float(note["start_time"])
            (
                snapped_time,
                snapped_divisor,
                min_error,
            ) = self._snap_time_to_grid(
                start_time,
                beat_duration_ms,
                first_beat_ms,
                self.config.BEAT_DIVISORS,
            )

            if min_error <= self.config.MAX_ALIGN_ERROR_MS:
                note["aligned_time"] = snapped_time
                note["snap_divisor"] = snapped_divisor
                valid_count += 1
            else:
                note["aligned_time"] = start_time
                note["snap_divisor"] = 0

            note["end_time"] = note["aligned_time"] + float(note["duration"])
            aligned_notes.append(note)

        print(f"对齐完成: {valid_count}/{len(note_events)} 个音符成功吸附到网格")
        return aligned_notes

    def apply_timing_grid_filter(self, aligned_notes, bpm, first_beat_time=0):
        print("应用时间点对齐过滤器...")

        if not aligned_notes:
            return []

        if not self.config.ENABLE_TIMING_GRID_FILTER or bpm <= 0:
            return [dict(note) for note in aligned_notes]

        beat_duration_ms = 60000 / bpm
        first_beat_ms = first_beat_time * 1000
        hold_divisors = [
            divisor
            for divisor in self.config.BEAT_DIVISORS
            if divisor <= self.config.TIMING_FILTER_HOLD_MIN_DIVISOR
        ]
        if not hold_divisors:
            hold_divisors = list(self.config.BEAT_DIVISORS)

        filtered_notes = []
        adjusted_count = 0

        for source_note in aligned_notes:
            note = dict(source_note)
            original_start = float(note.get("aligned_time", note.get("start_time", 0.0)))
            (
                snapped_start,
                snapped_divisor,
                _,
            ) = self._snap_time_to_grid(
                original_start,
                beat_duration_ms,
                first_beat_ms,
                self.config.BEAT_DIVISORS,
            )
            if abs(snapped_start - original_start) > 0.5:
                adjusted_count += 1

            note["aligned_time"] = snapped_start
            note["snap_divisor"] = snapped_divisor

            original_end = float(note.get("end_time", original_start + note.get("duration", 0.0)))
            if original_end > original_start:
                snapped_end, _, _ = self._snap_time_to_grid(
                    original_end,
                    beat_duration_ms,
                    first_beat_ms,
                    hold_divisors,
                )
                if snapped_end < snapped_start:
                    snapped_end = snapped_start
                note["duration"] = max(0.0, snapped_end - snapped_start)
                note["end_time"] = snapped_start + note["duration"]
            else:
                note["duration"] = max(0.0, float(note.get("duration", 0.0)))
                note["end_time"] = snapped_start + note["duration"]

            filtered_notes.append(note)

        print(f"时间点对齐过滤: 调整了 {adjusted_count}/{len(aligned_notes)} 个音符")
        return filtered_notes

    def apply_dynamic_density_filter(
        self, aligned_notes, energy_profile, hop_length, sr, beat_duration_ms
    ):
        print("应用动态密度控制...")

        if not aligned_notes:
            return []

        filtered_notes = []
        window_size_ms = max(125, int(self.config.SILENCE_WINDOW_MS))
        current_window_start = 0
        window_notes = []
        sorted_notes = sorted(aligned_notes, key=lambda x: x["aligned_time"])

        def process_window(notes, start_ms):
            if not notes:
                return []

            mid_time_ms = start_ms + window_size_ms / 2
            frame_idx = int((mid_time_ms / 1000) * sr / hop_length)
            frame_idx = min(max(frame_idx, 0), len(energy_profile) - 1)
            local_energy = float(energy_profile[frame_idx])

            allowed_snaps = [1]
            for threshold, snaps in self.config.SNAP_RESTRICTIONS:
                if local_energy >= threshold:
                    allowed_snaps = snaps
                else:
                    break

            valid_snap_notes = []
            for note in notes:
                divisor = note.get("snap_divisor", 0)
                if divisor in allowed_snaps or divisor == 1:
                    valid_snap_notes.append(note)

            if not valid_snap_notes:
                return []

            max_nps = 4.0
            for threshold, nps in self.config.DENSITY_MAPPING:
                if local_energy >= threshold:
                    max_nps = nps
                else:
                    break

            max_nps *= self.config.DENSITY_NPS_SCALE
            density_cap = max(1, int(round(max_nps * (window_size_ms / 1000))))

            if beat_duration_ms > 0:
                beat_cap = max(
                    1,
                    int(
                        round(
                            self.config.MAX_NOTES_PER_BEAT
                            * (window_size_ms / beat_duration_ms)
                        )
                    ),
                )
            else:
                beat_cap = len(valid_snap_notes)

            keep_ratio = min(max(self.config.DENSITY_FILTER_RATIO, 0.05), 1.0)
            ratio_cap = max(1, int(np.ceil(len(valid_snap_notes) * keep_ratio)))
            final_cap = min(density_cap, beat_cap, ratio_cap)

            if len(valid_snap_notes) > final_cap:
                valid_snap_notes.sort(
                    key=lambda x: (x.get("magnitude", 0), x["duration"]),
                    reverse=True,
                )
                valid_snap_notes = valid_snap_notes[:final_cap]

            return valid_snap_notes

        for note in sorted_notes:
            if note["aligned_time"] > current_window_start + window_size_ms:
                filtered_notes.extend(process_window(window_notes, current_window_start))
                window_notes = []
                current_window_start += window_size_ms
                while note["aligned_time"] > current_window_start + window_size_ms:
                    current_window_start += window_size_ms
            window_notes.append(note)

        filtered_notes.extend(process_window(window_notes, current_window_start))
        filtered_notes.sort(key=lambda x: x["aligned_time"])

        print(
            f"动态密度过滤: {len(aligned_notes)} -> {len(filtered_notes)} "
            f"(保留率 {len(filtered_notes)/len(aligned_notes):.1%})"
        )
        return filtered_notes

    def _moving_average(self, values, window_size):
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return np.zeros(1, dtype=np.float32)
        if window_size <= 1:
            return array.copy()
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        return np.convolve(array, kernel, mode="same").astype(np.float32)

    def _build_energy_context(self, energy_profile, hop_length, sr):
        short_env = np.asarray(energy_profile, dtype=np.float32)
        if short_env.size == 0:
            short_env = np.zeros(1, dtype=np.float32)

        long_window_frames = max(
            1,
            int(
                round(
                    (self.config.SILENCE_WINDOW_MS / 1000.0)
                    * (float(sr) / float(hop_length))
                    * 4.0
                )
            ),
        )
        long_env = self._moving_average(short_env, long_window_frames)
        contrast = short_env / np.maximum(long_env, 1e-4)
        times_ms = np.arange(short_env.size, dtype=np.float32) * float(hop_length) / float(sr) * 1000.0

        return {
            "times_ms": times_ms,
            "short_env": short_env,
            "long_env": long_env,
            "contrast": contrast,
        }

    def _sample_context_value(self, times_ms, values, time_ms):
        if times_ms.size == 0:
            return 0.0
        index = int(np.searchsorted(times_ms, float(time_ms), side="left"))
        index = max(0, min(index, times_ms.size - 1))
        return float(values[index])

    def _slice_context_values(self, times_ms, values, start_ms, end_ms):
        if times_ms.size == 0:
            return np.zeros(0, dtype=np.float32)
        start_index = int(np.searchsorted(times_ms, float(start_ms), side="left"))
        end_index = int(np.searchsorted(times_ms, float(end_ms), side="right"))
        start_index = max(0, min(start_index, times_ms.size))
        end_index = max(start_index, min(end_index, times_ms.size))
        return values[start_index:end_index]

    def _detect_leading_active_start_ms(self, energy_context):
        times_ms = energy_context["times_ms"]
        short_env = energy_context["short_env"]
        contrast = energy_context["contrast"]

        for index in range(times_ms.size):
            if (
                short_env[index] >= self.config.SILENCE_ONSET_ABS_THRESHOLD
                or contrast[index] >= self.config.SILENCE_ONSET_REL_THRESHOLD
            ):
                return max(0, int(round(times_ms[index])) - self.config.SILENCE_LEADING_MARGIN_MS)

        return 0

    def _window_is_quiet(self, energy_context, window_start_ms, window_end_ms):
        times_ms = energy_context["times_ms"]
        short_env = energy_context["short_env"]
        contrast = energy_context["contrast"]

        window_energy = self._slice_context_values(
            times_ms,
            short_env,
            window_start_ms,
            window_end_ms,
        )
        window_contrast = self._slice_context_values(
            times_ms,
            contrast,
            window_start_ms,
            window_end_ms,
        )

        if window_energy.size == 0:
            return True

        energy_peak = float(np.max(window_energy))
        energy_mean = float(np.mean(window_energy))
        contrast_peak = float(np.max(window_contrast)) if window_contrast.size > 0 else 0.0

        return (
            energy_peak < self.config.SILENCE_ABS_THRESHOLD
            and contrast_peak < self.config.SILENCE_REL_THRESHOLD
            and energy_mean < self.config.SILENCE_ABS_THRESHOLD * 1.05
        )

    def apply_silence_energy_filter(self, note_events, energy_profile, hop_length, sr):
        print("应用静音检测过滤器...")

        if not note_events:
            return []

        if not self.config.ENABLE_SILENCE_ENERGY_FILTER:
            return list(note_events)

        energy_context = self._build_energy_context(energy_profile, hop_length, sr)
        leading_start_ms = self._detect_leading_active_start_ms(energy_context)
        window_size_ms = max(125, int(self.config.SILENCE_WINDOW_MS))

        filtered_notes = []
        removed_intro = 0
        removed_quiet = 0

        for note in sorted(note_events, key=lambda item: item["aligned_time"]):
            note_time_ms = int(round(float(note["aligned_time"])))
            if note_time_ms < leading_start_ms:
                removed_intro += 1
                continue

            window_start_ms = (note_time_ms // window_size_ms) * window_size_ms
            window_end_ms = window_start_ms + window_size_ms
            if self._window_is_quiet(energy_context, window_start_ms, window_end_ms):
                removed_quiet += 1
                continue

            filtered_notes.append(note)

        print(
            f"静音检测过滤: {len(note_events)} -> {len(filtered_notes)} "
            f"(删除前导静音 {removed_intro}, 删除低能量窗口 {removed_quiet})"
        )
        return filtered_notes

    def map_frequency_to_column(self, note_events, num_columns=7):
        print(f"频率到轨道映射 ({num_columns}K)...")

        if num_columns not in self.config.COLUMN_MAPPING:
            raise ValueError(f"不支持的键数: {num_columns}K")

        column_mapping = self.config.COLUMN_MAPPING[num_columns]
        n_mels = self.config.N_MELS
        mapped_notes = []

        for source_note in note_events:
            note = dict(source_note)
            freq_bin = note["frequency_bin"]
            pitch_class = int((freq_bin / n_mels) * 12) % 12

            if pitch_class in column_mapping:
                column = column_mapping.index(pitch_class)
            else:
                distances = [abs(pitch_class - pc) for pc in column_mapping]
                column = int(np.argmin(distances))

            note["pitch_class"] = pitch_class
            note["column"] = column
            note["x_position"] = self._calculate_x_position(column, num_columns)
            mapped_notes.append(note)

        mapped_notes.sort(key=lambda item: item["aligned_time"])
        return mapped_notes

    def _rebalance_window_columns(self, window_notes, num_columns):
        if not window_notes:
            return 0

        counts = {column: 0 for column in range(num_columns)}
        by_column = {column: [] for column in range(num_columns)}
        for note in window_notes:
            counts[note["column"]] += 1
            by_column[note["column"]].append(note)

        target_max = max(1, int(np.ceil(len(window_notes) * self.config.COLUMN_BALANCE_MAX_SHARE)))
        if len(window_notes) <= num_columns or max(counts.values()) <= target_max:
            return 0

        for notes in by_column.values():
            notes.sort(key=lambda item: (item.get("magnitude", 0.0), item["aligned_time"]))

        adjusted_count = 0
        while True:
            overloaded_columns = [
                column for column in range(num_columns) if counts[column] > target_max
            ]
            if not overloaded_columns:
                break

            over_column = max(overloaded_columns, key=lambda column: counts[column])
            target_candidates = [
                column
                for column in range(num_columns)
                if column != over_column and counts[column] < target_max
            ]
            if not target_candidates or not by_column[over_column]:
                break

            note = by_column[over_column].pop(0)
            target_column = min(
                target_candidates,
                key=lambda column: (counts[column], abs(column - over_column)),
            )

            counts[over_column] -= 1
            counts[target_column] += 1
            note["column"] = target_column
            note["x_position"] = self._calculate_x_position(target_column, num_columns)
            by_column[target_column].append(note)
            by_column[target_column].sort(
                key=lambda item: (item.get("magnitude", 0.0), item["aligned_time"])
            )
            adjusted_count += 1

        return adjusted_count

    def apply_column_balance_filter(self, mapped_notes, num_columns):
        print("应用轨道均衡过滤器...")

        if not mapped_notes:
            return []

        if not self.config.ENABLE_COLUMN_BALANCE_FILTER:
            return list(mapped_notes)

        window_ms = max(250, int(self.config.COLUMN_BALANCE_WINDOW_MS))
        notes_by_window = {}
        for note in mapped_notes:
            window_index = int(note["aligned_time"] // window_ms)
            notes_by_window.setdefault(window_index, []).append(note)

        balanced_notes = []
        adjusted_count = 0
        for window_index in sorted(notes_by_window):
            window_notes = notes_by_window[window_index]
            adjusted_count += self._rebalance_window_columns(window_notes, num_columns)
            balanced_notes.extend(window_notes)

        balanced_notes.sort(key=lambda item: item["aligned_time"])
        print(f"轨道均衡过滤: 调整了 {adjusted_count} 个音符")
        return balanced_notes

    def enforce_physical_limits(self, mapped_notes):
        print("执行物理手感修正 (Gap Enforcement)...")

        if not mapped_notes:
            return []

        columns_notes = {}
        for note in mapped_notes:
            columns_notes.setdefault(note["column"], []).append(note)

        final_notes = []
        min_gap = max(
            self.config.MIN_COLUMN_GAP_MS,
            int(
                round(
                    self.config.MAX_SAME_COLUMN_INTERVAL_MS
                    * self.config.PHYSICAL_CORRECTION_STRICTNESS
                )
            ),
        )

        for notes in columns_notes.values():
            notes.sort(key=lambda x: x["aligned_time"])
            processed = []

            for note in notes:
                if not processed:
                    processed.append(note)
                    continue

                previous = processed[-1]
                gap = note["aligned_time"] - previous["end_time"]

                if gap >= min_gap:
                    processed.append(note)
                    continue

                if previous["duration"] >= self.config.HOLD_NOTE_MIN_DURATION:
                    new_end_time = note["aligned_time"] - min_gap
                    new_duration = new_end_time - previous["aligned_time"]

                    if new_duration >= self.config.HOLD_NOTE_MIN_DURATION:
                        previous["end_time"] = new_end_time
                        previous["duration"] = min(
                            new_duration, self.config.HOLD_NOTE_MAX_DURATION
                        )
                        processed.append(note)
                        continue

                processed[-1] = note

            final_notes.extend(processed)

        final_notes.sort(key=lambda x: x["aligned_time"])
        print(
            f"物理修正后: {len(final_notes)}/{len(mapped_notes)} 个音符 "
            f"(删除了 {len(mapped_notes) - len(final_notes)} 个冲突)"
        )
        return final_notes

    def normalize_hold_notes(self, notes):
        print("调整长条比例...")

        if not notes:
            return []

        normalized_notes = []
        candidates = []

        for source_note in notes:
            note = dict(source_note)
            note["duration"] = max(
                0,
                min(
                    int(round(note.get("duration", 0))),
                    self.config.HOLD_NOTE_MAX_DURATION,
                ),
            )
            note["end_time"] = note["aligned_time"] + note["duration"]
            normalized_notes.append(note)

            if note["duration"] >= self.config.HOLD_NOTE_MIN_DURATION:
                candidates.append(note)

        target_holds = int(
            round(len(normalized_notes) * self.config.HOLD_NOTE_TARGET_PERCENTAGE / 100)
        )
        candidates.sort(
            key=lambda item: (item["duration"], item.get("magnitude", 0)),
            reverse=True,
        )
        hold_ids = {id(note) for note in candidates[:target_holds]}

        hold_count = 0
        for note in normalized_notes:
            if id(note) in hold_ids:
                hold_count += 1
                note["duration"] = max(note["duration"], self.config.HOLD_NOTE_MIN_DURATION)
                note["duration"] = min(note["duration"], self.config.HOLD_NOTE_MAX_DURATION)
                note["end_time"] = note["aligned_time"] + note["duration"]
            else:
                note["duration"] = 0
                note["end_time"] = note["aligned_time"]

        actual_ratio = (hold_count / len(normalized_notes) * 100) if normalized_notes else 0
        print(f"长条调整后: {hold_count}/{len(normalized_notes)} ({actual_ratio:.1f}%)")
        return normalized_notes

    def _calculate_x_position(self, column, num_columns):
        width = 512 / num_columns
        return int(width * (column + 0.5))

    def extract_features(self, audio_data, note_events):
        print("\n=== 特征提取开始 ===")

        bpm_info = self.detect_bpm(audio_data["audio"], audio_data["sample_rate"])
        beat_duration_ms = 60000 / bpm_info["bpm"] if bpm_info["bpm"] else 0

        aligned_notes = self.align_to_beat_grid(
            note_events, bpm_info["bpm"], bpm_info["first_beat_time"]
        )
        timing_filtered_notes = self.apply_timing_grid_filter(
            aligned_notes,
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
        )
        density_filtered_notes = self.apply_dynamic_density_filter(
            timing_filtered_notes,
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
            beat_duration_ms,
        )
        silence_filtered_notes = self.apply_silence_energy_filter(
            density_filtered_notes,
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
        )
        mapped_notes = self.map_frequency_to_column(
            silence_filtered_notes, self.config.DEFAULT_COLUMNS
        )
        balanced_notes = self.apply_column_balance_filter(
            mapped_notes, self.config.DEFAULT_COLUMNS
        )
        physically_correct_notes = self.enforce_physical_limits(balanced_notes)
        final_notes = self.normalize_hold_notes(physically_correct_notes)

        print("=== 特征提取完成 ===\n")

        return {
            "bpm_info": bpm_info,
            "aligned_notes": aligned_notes,
            "timing_filtered_notes": timing_filtered_notes,
            "density_filtered_notes": density_filtered_notes,
            "silence_filtered_notes": silence_filtered_notes,
            "mapped_notes": mapped_notes,
            "balanced_notes": balanced_notes,
            "controlled_notes": final_notes,
            "config": {
                "columns": self.config.DEFAULT_COLUMNS,
                "sample_rate": audio_data["sample_rate"],
                "hop_length": audio_data["hop_length"],
            },
        }
