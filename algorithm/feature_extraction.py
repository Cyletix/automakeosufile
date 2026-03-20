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

    def _normalize_profile(self, values):
        profile = np.asarray(values, dtype=np.float32)
        if profile.size == 0:
            return np.zeros(1, dtype=np.float32)
        profile = np.maximum(profile, 0.0)
        profile_min = float(np.min(profile))
        profile_max = float(np.max(profile))
        if profile_max > profile_min:
            profile = (profile - profile_min) / (profile_max - profile_min)
        else:
            profile = np.zeros_like(profile)
        return profile.astype(np.float32)

    def detect_bpm(self, y, sr, onset_profile=None):
        print("检测BPM...")
        frame_length = self.config.N_FFT
        hop_length = self.config.HOP_LENGTH

        if onset_profile is not None:
            onset_envelope = self._normalize_profile(onset_profile)
        elif len(y) < frame_length:
            bpm = 120.0
            beats = np.array([0.0])
            first_beat_time = 0.0
            print(f"检测到BPM: {bpm:.1f}, 第一个节拍: {first_beat_time:.2f}s")
            return {"bpm": bpm, "first_beat_time": first_beat_time, "beats": beats}
        else:
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

    def build_dynamic_beat_path(self, onset_profile, hop_length, sr, bpm, first_beat_time=0):
        onset = self._normalize_profile(onset_profile)
        if onset.size == 0 or bpm <= 0:
            return {
                "beat_times_ms": [],
                "phase_offset_ms": float(first_beat_time) * 1000.0,
                "confidence": 0.0,
            }

        beat_step_frames = max(1, int(round((60.0 / float(bpm)) * float(sr) / float(hop_length))))
        phase_stride = max(1, beat_step_frames // 24)
        best_phase = 0
        best_score = -1.0

        for phase in range(0, beat_step_frames, phase_stride):
            score = 0.0
            for frame_idx in range(phase, onset.size, beat_step_frames):
                start_idx = max(0, frame_idx - 1)
                end_idx = min(onset.size, frame_idx + 2)
                score += float(np.max(onset[start_idx:end_idx]))
            if score > best_score:
                best_score = score
                best_phase = phase

        beat_times_ms = []
        for frame_idx in range(best_phase, onset.size, beat_step_frames):
            start_idx = max(0, frame_idx - 1)
            end_idx = min(onset.size, frame_idx + 2)
            local_window = onset[start_idx:end_idx]
            if local_window.size == 0:
                continue
            local_peak_offset = int(np.argmax(local_window))
            refined_frame = start_idx + local_peak_offset
            beat_times_ms.append(float(refined_frame) * float(hop_length) / float(sr) * 1000.0)

        phase_offset_ms = float(best_phase) * float(hop_length) / float(sr) * 1000.0
        confidence = 0.0
        if onset.size > 0 and beat_times_ms:
            confidence = best_score / max(1.0, len(beat_times_ms))

        return {
            "beat_times_ms": beat_times_ms,
            "phase_offset_ms": phase_offset_ms,
            "confidence": confidence,
        }

    def _sample_profile_activity(self, energy_profile, hop_length, sr, time_ms):
        profile = np.asarray(energy_profile, dtype=np.float32)
        if profile.size == 0:
            return 0.0
        frame_idx = int((float(time_ms) / 1000.0) * float(sr) / float(hop_length))
        frame_idx = min(max(frame_idx, 0), profile.size - 1)
        start_idx = max(0, frame_idx - 1)
        end_idx = min(profile.size, frame_idx + 2)
        return float(np.max(profile[start_idx:end_idx]))

    def _normalize_slot_vector(self, slot_values):
        vector = np.asarray(slot_values, dtype=np.float32)
        if vector.size == 0:
            return vector
        max_value = float(np.max(vector))
        if max_value > 1e-6:
            vector = vector / max_value
        return vector

    def _slot_pattern_similarity(self, left_slot_values, right_slot_values):
        left_vector = self._normalize_slot_vector(left_slot_values)
        right_vector = self._normalize_slot_vector(right_slot_values)
        if left_vector.size == 0 or right_vector.size == 0:
            return 0.0
        size = min(left_vector.size, right_vector.size)
        left_vector = left_vector[:size]
        right_vector = right_vector[:size]
        left_norm = float(np.linalg.norm(left_vector))
        right_norm = float(np.linalg.norm(right_vector))
        if left_norm <= 1e-6 or right_norm <= 1e-6:
            return 0.0
        similarity = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
        if np.isnan(similarity):
            return 0.0
        return max(0.0, min(1.0, similarity))

    def _build_divisor_profile(self, onset_profiles, energy_profile, divisor):
        base_profile = self._normalize_profile(energy_profile)
        if not isinstance(onset_profiles, dict) or not onset_profiles:
            return base_profile

        low_profile = self._normalize_profile(onset_profiles.get("low", base_profile))
        mid_profile = self._normalize_profile(onset_profiles.get("mid", base_profile))
        high_profile = self._normalize_profile(onset_profiles.get("high", base_profile))
        combined_profile = self._normalize_profile(onset_profiles.get("combined", base_profile))

        if divisor <= 2:
            profile = low_profile * 0.45 + mid_profile * 0.35 + combined_profile * 0.20
        elif divisor <= 4:
            profile = mid_profile * 0.42 + high_profile * 0.28 + combined_profile * 0.30
        else:
            profile = high_profile * 0.50 + mid_profile * 0.20 + combined_profile * 0.30
        return self._normalize_profile(profile)

    def build_transient_layer_metrics(
        self, energy_profile, hop_length, sr, bpm, first_beat_time=0, onset_profiles=None
    ):
        metrics = {}
        weights = {}
        profile = np.asarray(energy_profile, dtype=np.float32)
        if profile.size == 0 or bpm <= 0:
            return metrics, weights

        beat_duration_ms = 60000.0 / float(bpm)
        first_beat_ms = float(first_beat_time) * 1000.0
        total_duration_ms = (
            float(profile.size - 1) * float(hop_length) / float(sr) * 1000.0
        )
        raw_scores = {}

        for divisor in self.config.TRANSIENT_LAYER_DIVISORS:
            step_ms = beat_duration_ms / float(divisor)
            if step_ms <= 0:
                continue
            divisor_profile = self._build_divisor_profile(onset_profiles, energy_profile, divisor)
            base_diff = np.maximum(0.0, np.diff(divisor_profile, prepend=divisor_profile[0]))

            positions_ms = np.arange(first_beat_ms, total_duration_ms + step_ms, step_ms)
            if positions_ms.size == 0:
                continue

            diff_samples = []
            peak_samples = []
            for position_ms in positions_ms:
                frame_idx = int((float(position_ms) / 1000.0) * float(sr) / float(hop_length))
                frame_idx = min(max(frame_idx, 0), divisor_profile.size - 1)
                diff_samples.append(float(base_diff[frame_idx]))
                peak_samples.append(self._sample_profile_activity(divisor_profile, hop_length, sr, position_ms))

            diff_samples = np.asarray(diff_samples, dtype=np.float32)
            peak_samples = np.asarray(peak_samples, dtype=np.float32)
            if diff_samples.size == 0:
                continue

            diff_mean = float(np.mean(diff_samples))
            diff_std = float(np.std(diff_samples))
            peak_mean = float(np.mean(peak_samples))
            peak_std = float(np.std(peak_samples))

            diff_threshold = diff_mean + diff_std * 0.35
            peak_threshold = peak_mean + peak_std * 0.35
            diff_recurrence = float(np.count_nonzero(diff_samples >= diff_threshold)) / float(diff_samples.size)
            peak_recurrence = float(np.count_nonzero(peak_samples >= peak_threshold)) / float(peak_samples.size)

            weighted_score = (
                float(self.config.TRANSIENT_DIFF_WEIGHTS.get(int(divisor), 0.2))
                * (diff_mean * float(self.config.TRANSIENT_GRADIENT_WEIGHT) + peak_mean * float(self.config.TRANSIENT_PEAK_WEIGHT))
                * float(self.config.TRANSIENT_LAYER_PRIORS.get(int(divisor), 0.2))
            )
            recurrence_score = (diff_recurrence * 0.6) + (peak_recurrence * 0.4)
            raw_score = max(0.0, weighted_score) * max(0.0, recurrence_score)

            metrics[int(divisor)] = {
                'diff_mean': diff_mean,
                'diff_recurrence': diff_recurrence,
                'peak_mean': peak_mean,
                'peak_recurrence': peak_recurrence,
                'raw_score': raw_score,
            }
            raw_scores[int(divisor)] = raw_score

        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 1e-6:
            for divisor in raw_scores.keys():
                weights[int(divisor)] = float(self.config.TRANSIENT_LAYER_PRIORS.get(int(divisor), 0.2))
        else:
            for divisor, raw_score in raw_scores.items():
                normalized = max(0.05, float(raw_score) / float(max_score))
                weights[int(divisor)] = float(normalized ** float(self.config.TRANSIENT_POWER))

        return metrics, weights

    def build_divisor_weight_metrics(
        self, energy_profile, hop_length, sr, bpm, first_beat_time=0, onset_profile=None
    ):
        metrics = {}
        weights = {}
        source_profile = onset_profile if onset_profile is not None else energy_profile
        profile = np.asarray(source_profile, dtype=np.float32)
        if profile.size == 0 or bpm <= 0:
            return metrics, weights

        beat_duration_ms = 60000.0 / float(bpm)
        first_beat_ms = float(first_beat_time) * 1000.0
        total_duration_ms = (
            float(profile.size - 1) * float(hop_length) / float(sr) * 1000.0
        )
        raw_scores = {}

        for divisor in self.config.BEAT_DIVISORS:
            step_ms = beat_duration_ms / float(divisor)
            if step_ms <= 0:
                continue
            positions_ms = np.arange(first_beat_ms, total_duration_ms + step_ms, step_ms)
            if positions_ms.size == 0:
                continue

            activities = np.array(
                [
                    self._sample_profile_activity(profile, hop_length, sr, position_ms)
                    for position_ms in positions_ms
                ],
                dtype=np.float32,
            )
            if activities.size == 0:
                continue

            mean_energy = float(np.mean(activities))
            std_energy = float(np.std(activities))
            top_count = max(1, int(round(activities.size * 0.2)))
            strongest_energy = float(
                np.mean(np.sort(activities)[-top_count:])
            )
            recurrence_threshold = mean_energy + std_energy * 0.35
            recurrence_ratio = float(
                np.count_nonzero(activities >= recurrence_threshold)
            ) / float(activities.size)
            prior = float(self.config.GRID_WEIGHT_PRIORS.get(divisor, 0.2))
            raw_score = prior * (
                strongest_energy * float(self.config.GRID_WEIGHT_ENERGY_WEIGHT)
                + recurrence_ratio * float(self.config.GRID_WEIGHT_RECURRENCE_WEIGHT)
            )
            metrics[int(divisor)] = {
                "strongest_energy": strongest_energy,
                "recurrence_ratio": recurrence_ratio,
                "mean_energy": mean_energy,
                "prior": prior,
                "raw_score": raw_score,
            }
            raw_scores[int(divisor)] = raw_score

        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 1e-6:
            for divisor in raw_scores.keys():
                weights[int(divisor)] = float(self.config.GRID_WEIGHT_PRIORS.get(divisor, 0.2))
        else:
            for divisor, raw_score in raw_scores.items():
                weights[int(divisor)] = max(0.05, float(raw_score) / float(max_score))

        return metrics, weights

    def _classify_beat_family(self, divisor):
        divisor = int(divisor)
        if divisor in self.config.BEAT_FAMILY_TRIPLET_DIVISORS:
            return "triplet"
        if divisor in self.config.BEAT_FAMILY_BINARY_DIVISORS:
            return "binary"
        return "neutral"

    def _build_local_beat_family_scores(
        self,
        energy_profile,
        hop_length,
        sr,
        beat_duration_ms,
        first_beat_ms,
        window_start_ms,
        window_end_ms,
        onset_profiles=None,
        divisor_weights=None,
    ):
        family_scores = {
            "binary": 0.0,
            "triplet": 0.0,
        }
        family_detail = {}

        for divisor in self.config.BEAT_DIVISORS:
            if divisor <= 0:
                continue

            family = self._classify_beat_family(divisor)
            if family == "neutral":
                continue

            step_ms = beat_duration_ms / float(divisor)
            if step_ms <= 0:
                continue

            divisor_profile = self._build_divisor_profile(onset_profiles, energy_profile, divisor)
            positions_ms = np.arange(
                max(float(window_start_ms), float(first_beat_ms)),
                float(window_end_ms) + step_ms * 0.5,
                step_ms,
            )
            if positions_ms.size == 0:
                family_detail[int(divisor)] = 0.0
                continue

            activities = np.array(
                [
                    self._sample_profile_activity(divisor_profile, hop_length, sr, position_ms)
                    for position_ms in positions_ms
                ],
                dtype=np.float32,
            )
            if activities.size == 0:
                family_detail[int(divisor)] = 0.0
                continue

            mean_activity = float(np.mean(activities))
            std_activity = float(np.std(activities))
            strongest_count = max(1, int(round(activities.size * 0.25)))
            strongest_activity = float(np.mean(np.sort(activities)[-strongest_count:]))
            recurrence_threshold = mean_activity + std_activity * 0.20
            recurrence_ratio = float(np.count_nonzero(activities >= recurrence_threshold)) / float(activities.size)
            global_weight = float(
                (divisor_weights or {}).get(int(divisor), self.config.GRID_WEIGHT_PRIORS.get(int(divisor), 0.2))
            )
            divisor_score = global_weight * (
                strongest_activity * float(self.config.SECTION_STATE_LOCAL_ENERGY_WEIGHT)
                + recurrence_ratio * float(self.config.SECTION_STATE_LOCAL_REPETITION_WEIGHT)
            )
            family_detail[int(divisor)] = max(0.0, float(divisor_score))
            family_scores[family] += family_detail[int(divisor)]

        family_scores["binary"] *= float(self.config.BEAT_FAMILY_BINARY_PRIOR)
        family_scores["triplet"] *= float(self.config.BEAT_FAMILY_TRIPLET_PRIOR)
        return family_scores, family_detail

    def build_beat_family_state(
        self,
        energy_profile,
        hop_length,
        sr,
        bpm,
        first_beat_time=0,
        onset_profiles=None,
        divisor_weights=None,
    ):
        profile = np.asarray(energy_profile, dtype=np.float32)
        if profile.size == 0 or bpm <= 0:
            return {
                "window_ms": 0.0,
                "first_window_start_ms": 0.0,
                "windows": [],
                "global_family": "neutral",
            }

        beat_duration_ms = 60000.0 / float(bpm)
        window_beats = max(2, int(self.config.BEAT_FAMILY_WINDOW_BEATS))
        window_ms = beat_duration_ms * float(window_beats)
        total_duration_ms = float(profile.size - 1) * float(hop_length) / float(sr) * 1000.0
        first_beat_ms = float(first_beat_time) * 1000.0
        windows = []
        window_start_ms = 0.0

        while window_start_ms <= total_duration_ms + 1e-3:
            window_end_ms = window_start_ms + window_ms
            local_family_scores, family_detail = self._build_local_beat_family_scores(
                energy_profile,
                hop_length,
                sr,
                beat_duration_ms,
                first_beat_ms,
                window_start_ms,
                window_end_ms,
                onset_profiles,
                divisor_weights,
            )

            binary_score = float(local_family_scores.get("binary", 0.0))
            triplet_score = float(local_family_scores.get("triplet", 0.0))
            strongest = max(binary_score, triplet_score, 1e-6)
            balance = 1.0 - min(1.0, abs(binary_score - triplet_score) / strongest)
            neutral_score = strongest * max(0.18, balance)

            windows.append(
                {
                    "start_ms": float(window_start_ms),
                    "end_ms": float(window_end_ms),
                    "family_scores": {
                        "binary": binary_score,
                        "triplet": triplet_score,
                        "neutral": float(neutral_score),
                    },
                    "family_detail": family_detail,
                }
            )
            window_start_ms += window_ms

        if not windows:
            return {
                "window_ms": float(window_ms),
                "first_window_start_ms": 0.0,
                "windows": [],
                "global_family": "neutral",
            }

        family_states = ["binary", "neutral", "triplet"]
        state_count = len(family_states)
        window_count = len(windows)
        dp = np.full((window_count, state_count), -np.inf, dtype=np.float32)
        backtrack = np.full((window_count, state_count), -1, dtype=np.int32)
        stay_bonus = float(self.config.BEAT_FAMILY_STAY_BONUS)
        switch_penalty = float(self.config.BEAT_FAMILY_SWITCH_PENALTY)

        for state_index, family in enumerate(family_states):
            dp[0, state_index] = float(windows[0]["family_scores"].get(family, 0.0))

        for window_index in range(1, window_count):
            for state_index, family in enumerate(family_states):
                local_score = float(windows[window_index]["family_scores"].get(family, 0.0))
                best_total = -np.inf
                best_prev = -1
                for prev_index, prev_family in enumerate(family_states):
                    if not np.isfinite(dp[window_index - 1, prev_index]):
                        continue
                    transition_penalty = -stay_bonus if prev_family == family else switch_penalty
                    candidate_total = dp[window_index - 1, prev_index] + local_score - transition_penalty
                    if candidate_total > best_total:
                        best_total = candidate_total
                        best_prev = prev_index
                dp[window_index, state_index] = best_total
                backtrack[window_index, state_index] = best_prev

        final_state_index = int(np.argmax(dp[-1]))
        chosen_families = [family_states[final_state_index]]
        for window_index in range(window_count - 1, 0, -1):
            final_state_index = int(backtrack[window_index, final_state_index])
            if final_state_index < 0:
                final_state_index = 1
            chosen_families.append(family_states[final_state_index])
        chosen_families.reverse()

        total_binary = sum(float(window["family_scores"]["binary"]) for window in windows)
        total_triplet = sum(float(window["family_scores"]["triplet"]) for window in windows)
        global_family = "neutral"
        dominance_threshold = float(self.config.BEAT_FAMILY_GLOBAL_DOMINANCE_THRESHOLD)
        if total_binary > 1e-6 and total_binary >= total_triplet * dominance_threshold:
            global_family = "binary"
        elif total_triplet > 1e-6 and total_triplet >= total_binary * dominance_threshold:
            global_family = "triplet"

        for window, chosen_family in zip(windows, chosen_families):
            binary_score = float(window["family_scores"]["binary"])
            triplet_score = float(window["family_scores"]["triplet"])
            strongest = max(binary_score, triplet_score, 1e-6)
            score_gap_ratio = abs(binary_score - triplet_score) / strongest
            if chosen_family != "neutral" and score_gap_ratio < float(self.config.BEAT_FAMILY_NEUTRAL_MARGIN):
                chosen_family = "neutral"
            window["chosen_family"] = chosen_family
            window["score_gap_ratio"] = float(score_gap_ratio)

        stable_segments = []
        segment_start = windows[0]["start_ms"]
        segment_family = windows[0]["chosen_family"]
        for window in windows[1:]:
            if window["chosen_family"] != segment_family:
                stable_segments.append(
                    {
                        "start_ms": float(segment_start),
                        "end_ms": float(window["start_ms"]),
                        "family": str(segment_family),
                    }
                )
                segment_start = window["start_ms"]
                segment_family = window["chosen_family"]
        stable_segments.append(
            {
                "start_ms": float(segment_start),
                "end_ms": float(windows[-1]["end_ms"]),
                "family": str(segment_family),
            }
        )

        return {
            "window_ms": float(window_ms),
            "first_window_start_ms": 0.0,
            "windows": windows,
            "global_family": global_family,
            "stable_segments": stable_segments,
            "global_scores": {
                "binary": float(total_binary),
                "triplet": float(total_triplet),
            },
        }

    def _map_divisor_to_section_tier(self, divisor):
        state_divisors = sorted(int(value) for value in self.config.SECTION_STATE_DIVISORS)
        if not state_divisors:
            return int(divisor)
        for tier in state_divisors:
            if int(divisor) <= tier:
                return tier
        return state_divisors[-1]

    def _build_local_section_divisor_scores(
        self,
        energy_profile,
        hop_length,
        sr,
        beat_duration_ms,
        first_beat_ms,
        window_start_ms,
        window_end_ms,
        onset_profiles=None,
        divisor_weights=None,
    ):
        local_scores = {}
        state_divisors = sorted(int(value) for value in self.config.SECTION_STATE_DIVISORS)

        for divisor in state_divisors:
            if divisor <= 0:
                continue
            step_ms = beat_duration_ms / float(divisor)
            if step_ms <= 0:
                continue

            divisor_profile = self._build_divisor_profile(onset_profiles, energy_profile, divisor)
            positions_ms = np.arange(
                max(float(window_start_ms), float(first_beat_ms)),
                float(window_end_ms) + step_ms * 0.5,
                step_ms,
            )
            if positions_ms.size == 0:
                local_scores[int(divisor)] = 0.0
                continue

            activities = np.array(
                [
                    self._sample_profile_activity(divisor_profile, hop_length, sr, position_ms)
                    for position_ms in positions_ms
                ],
                dtype=np.float32,
            )
            if activities.size == 0:
                local_scores[int(divisor)] = 0.0
                continue

            mean_activity = float(np.mean(activities))
            std_activity = float(np.std(activities))
            top_count = max(1, int(round(activities.size * 0.25)))
            strongest_activity = float(np.mean(np.sort(activities)[-top_count:]))
            recurrence_threshold = mean_activity + std_activity * 0.20
            recurrence_ratio = float(np.count_nonzero(activities >= recurrence_threshold)) / float(activities.size)
            global_weight = float((divisor_weights or {}).get(int(divisor), self.config.GRID_WEIGHT_PRIORS.get(int(divisor), 0.2)))
            score = global_weight * (
                strongest_activity * float(self.config.SECTION_STATE_LOCAL_ENERGY_WEIGHT)
                + recurrence_ratio * float(self.config.SECTION_STATE_LOCAL_REPETITION_WEIGHT)
            )
            local_scores[int(divisor)] = max(0.0, float(score))

        return local_scores

    def build_section_divisor_state(
        self,
        energy_profile,
        hop_length,
        sr,
        bpm,
        first_beat_time=0,
        onset_profiles=None,
        divisor_weights=None,
    ):
        state_divisors = sorted(int(value) for value in self.config.SECTION_STATE_DIVISORS)
        profile = np.asarray(energy_profile, dtype=np.float32)
        if profile.size == 0 or bpm <= 0 or not state_divisors:
            return {
                "window_ms": 0.0,
                "first_window_start_ms": 0.0,
                "windows": [],
            }

        beat_duration_ms = 60000.0 / float(bpm)
        window_beats = max(2, int(self.config.SECTION_STATE_WINDOW_BEATS))
        window_ms = beat_duration_ms * float(window_beats)
        total_duration_ms = float(profile.size - 1) * float(hop_length) / float(sr) * 1000.0
        first_beat_ms = float(first_beat_time) * 1000.0
        windows = []
        window_start_ms = 0.0

        while window_start_ms <= total_duration_ms + 1e-3:
            window_end_ms = window_start_ms + window_ms
            local_scores = self._build_local_section_divisor_scores(
                energy_profile,
                hop_length,
                sr,
                beat_duration_ms,
                first_beat_ms,
                window_start_ms,
                window_end_ms,
                onset_profiles,
                divisor_weights,
            )

            state_scores = {}
            for state_divisor in state_divisors:
                allowed_score = 0.0
                suppressed_score = 0.0
                for divisor, local_score in local_scores.items():
                    if divisor <= state_divisor:
                        contribution = local_score
                        if divisor < state_divisor:
                            contribution *= float(self.config.SECTION_STATE_ALLOW_COARSER_WEIGHT)
                        allowed_score += contribution
                    else:
                        suppressed_score += local_score

                state_score = (
                    allowed_score
                    - suppressed_score * float(self.config.SECTION_STATE_FINE_DIVISOR_PENALTY)
                ) * float(self.config.SECTION_STATE_PRIORS.get(int(state_divisor), 0.2))
                state_scores[int(state_divisor)] = float(state_score)

            windows.append(
                {
                    "start_ms": float(window_start_ms),
                    "end_ms": float(window_end_ms),
                    "local_divisor_scores": local_scores,
                    "state_scores": state_scores,
                }
            )
            window_start_ms += window_ms

        if not windows:
            return {
                "window_ms": window_ms,
                "first_window_start_ms": 0.0,
                "windows": [],
            }

        state_count = len(state_divisors)
        window_count = len(windows)
        dp = np.full((window_count, state_count), -np.inf, dtype=np.float32)
        backtrack = np.full((window_count, state_count), -1, dtype=np.int32)
        stay_bonus = float(self.config.SECTION_STATE_STAY_BONUS)
        switch_penalty = float(self.config.SECTION_STATE_SWITCH_PENALTY)
        switch_distance_scale = float(self.config.SECTION_STATE_SWITCH_DISTANCE_SCALE)

        for state_index, state_divisor in enumerate(state_divisors):
            dp[0, state_index] = float(windows[0]["state_scores"].get(int(state_divisor), 0.0))

        for window_index in range(1, window_count):
            for state_index, state_divisor in enumerate(state_divisors):
                local_score = float(windows[window_index]["state_scores"].get(int(state_divisor), 0.0))
                best_total = -np.inf
                best_prev = -1
                for prev_index, prev_state_divisor in enumerate(state_divisors):
                    if not np.isfinite(dp[window_index - 1, prev_index]):
                        continue
                    distance = abs(np.log2(float(state_divisor)) - np.log2(float(prev_state_divisor)))
                    transition_penalty = 0.0
                    if prev_state_divisor != state_divisor:
                        transition_penalty = switch_penalty * (1.0 + distance * switch_distance_scale)
                    else:
                        transition_penalty = -stay_bonus
                    candidate_total = dp[window_index - 1, prev_index] + local_score - transition_penalty
                    if candidate_total > best_total:
                        best_total = candidate_total
                        best_prev = prev_index
                dp[window_index, state_index] = best_total
                backtrack[window_index, state_index] = best_prev

        final_state_index = int(np.argmax(dp[-1]))
        chosen_states = [state_divisors[final_state_index]]
        for window_index in range(window_count - 1, 0, -1):
            final_state_index = int(backtrack[window_index, final_state_index])
            if final_state_index < 0:
                final_state_index = 0
            chosen_states.append(state_divisors[final_state_index])
        chosen_states.reverse()

        for window, chosen_state in zip(windows, chosen_states):
            local_scores = window["local_divisor_scores"]
            max_local_score = max(local_scores.values(), default=0.0)
            if max_local_score <= 1e-6:
                normalized_scores = {
                    int(divisor): float(self.config.SECTION_STATE_PRIORS.get(int(divisor), 0.2))
                    for divisor in state_divisors
                }
            else:
                normalized_scores = {
                    int(divisor): max(0.05, float(score) / float(max_local_score))
                    for divisor, score in local_scores.items()
                }
            anchor_threshold = float(self.config.SECTION_STATE_ANCHOR_RATIO_THRESHOLD)
            anchor_divisors = [
                int(divisor)
                for divisor, score in normalized_scores.items()
                if int(divisor) <= int(chosen_state) and float(score) >= anchor_threshold
            ]
            if int(chosen_state) not in anchor_divisors:
                anchor_divisors.append(int(chosen_state))
            anchor_divisors = sorted(set(anchor_divisors))
            window["chosen_state_divisor"] = int(chosen_state)
            window["normalized_local_scores"] = normalized_scores
            window["chosen_state_score"] = float(window["state_scores"].get(int(chosen_state), 0.0))
            window["anchor_divisors"] = anchor_divisors

        stable_segments = []
        segment_start = windows[0]["start_ms"]
        segment_state = windows[0]["chosen_state_divisor"]
        for window in windows[1:]:
            if int(window["chosen_state_divisor"]) != int(segment_state):
                stable_segments.append(
                    {
                        "start_ms": float(segment_start),
                        "end_ms": float(window["start_ms"]),
                        "state_divisor": int(segment_state),
                    }
                )
                segment_start = window["start_ms"]
                segment_state = window["chosen_state_divisor"]
        stable_segments.append(
            {
                "start_ms": float(segment_start),
                "end_ms": float(windows[-1]["end_ms"]),
                "state_divisor": int(segment_state),
            }
        )

        return {
            "window_ms": float(window_ms),
            "first_window_start_ms": 0.0,
            "windows": windows,
            "stable_segments": stable_segments,
        }

    def _resolve_section_window(self, time_ms, section_divisor_state):
        if not section_divisor_state:
            return None
        windows = section_divisor_state.get("windows", [])
        if not windows:
            return None
        window_ms = float(section_divisor_state.get("window_ms", 0.0))
        if window_ms <= 0:
            return None
        first_window_start_ms = float(section_divisor_state.get("first_window_start_ms", 0.0))
        window_index = int((float(time_ms) - first_window_start_ms) // window_ms)
        window_index = max(0, min(window_index, len(windows) - 1))
        return windows[window_index]

    def _resolve_beat_family_window(self, time_ms, beat_family_state):
        if not beat_family_state:
            return None
        windows = beat_family_state.get("windows", [])
        if not windows:
            return None
        window_ms = float(beat_family_state.get("window_ms", 0.0))
        if window_ms <= 0:
            return None
        first_window_start_ms = float(beat_family_state.get("first_window_start_ms", 0.0))
        window_index = int((float(time_ms) - first_window_start_ms) // window_ms)
        window_index = max(0, min(window_index, len(windows) - 1))
        return windows[window_index]

    def _build_bar_section_novelty(self, bars):
        if not bars:
            return [], [], []

        profiles = [np.asarray(bar.get("slot_profile", []), dtype=np.float32) for bar in bars]
        audio_values = np.array(
            [float(np.mean(bar.get("slot_onsets", []))) for bar in bars],
            dtype=np.float32,
        )
        novelty = np.zeros(len(bars), dtype=np.float32)
        audio_novelty = np.zeros(len(bars), dtype=np.float32)
        radius = max(1, int(self.config.BAR_PATTERN_SECTION_NOVELTY_RADIUS))

        for bar_index in range(1, len(bars) - 1):
            left_start = max(0, bar_index - radius)
            right_end = min(len(bars), bar_index + 1 + radius)
            left_profiles = profiles[left_start:bar_index]
            right_profiles = profiles[bar_index + 1:right_end]
            if left_profiles and right_profiles:
                left_mean = np.mean(np.vstack(left_profiles), axis=0)
                right_mean = np.mean(np.vstack(right_profiles), axis=0)
                novelty[bar_index] = 1.0 - self._slot_pattern_similarity(left_mean, right_mean)

            audio_left = float(np.mean(audio_values[left_start:bar_index])) if bar_index > left_start else float(audio_values[max(0, bar_index - 1)])
            audio_right = float(np.mean(audio_values[bar_index + 1:right_end])) if right_end > bar_index + 1 else float(audio_values[bar_index])
            audio_novelty[bar_index] = abs(audio_right - audio_left)

        novelty = self._normalize_slot_vector(novelty)
        audio_novelty = self._normalize_slot_vector(audio_novelty)
        combined = self._normalize_slot_vector(
            novelty * float(self.config.BAR_PATTERN_SECTION_PATTERN_NOVELTY_WEIGHT)
            + audio_novelty * float(self.config.BAR_PATTERN_SECTION_AUDIO_NOVELTY_WEIGHT)
        )
        return novelty.tolist(), audio_novelty.tolist(), combined.tolist()

    def _segment_bar_sections(self, bars):
        if not bars:
            return [], [], [], []

        novelty, audio_novelty, combined = self._build_bar_section_novelty(bars)
        novelty_array = np.asarray(combined, dtype=np.float32)
        threshold = float(np.mean(novelty_array) + np.std(novelty_array) * float(self.config.BAR_PATTERN_SECTION_THRESHOLD_STD))
        min_section_bars = max(2, int(self.config.BAR_PATTERN_SECTION_MIN_BARS))
        boundaries = [0]

        for bar_index in range(1, len(bars) - 1):
            value = float(novelty_array[bar_index])
            if value < threshold:
                continue
            if value < float(novelty_array[bar_index - 1]) or value < float(novelty_array[bar_index + 1]):
                continue
            if bar_index - boundaries[-1] < min_section_bars:
                continue
            boundaries.append(int(bar_index))

        if len(bars) - boundaries[-1] < min_section_bars and len(boundaries) > 1:
            boundaries.pop()
        boundaries.append(len(bars))

        sections = []
        for section_index in range(len(boundaries) - 1):
            start_bar = int(boundaries[section_index])
            end_bar = int(boundaries[section_index + 1])
            if end_bar <= start_bar:
                continue
            sections.append(
                {
                    "section_index": int(section_index),
                    "start_bar_index": int(start_bar),
                    "end_bar_index": int(end_bar),
                    "start_ms": float(bars[start_bar]["start_ms"]),
                    "end_ms": float(bars[end_bar - 1]["end_ms"]),
                    "bar_count": int(end_bar - start_bar),
                    "novelty_peak": float(np.max(novelty_array[start_bar:end_bar])) if end_bar > start_bar else 0.0,
                }
            )

        return sections, novelty, audio_novelty, combined

    def _extract_section_bar_prototypes(self, bars, start_index, end_index):
        section_bars = bars[start_index:end_index]
        if not section_bars:
            return {"prototypes": [], "assignments": []}

        section_bar_count = len(section_bars)
        if section_bar_count < int(self.config.BAR_PATTERN_PROTOTYPE_MEDIUM_SECTION_BARS):
            prototype_count = 1
        elif section_bar_count < int(self.config.BAR_PATTERN_PROTOTYPE_LARGE_SECTION_BARS):
            prototype_count = 2
        else:
            prototype_count = int(self.config.BAR_PATTERN_PROTOTYPE_MAX_COUNT)

        profiles = [np.asarray(bar.get("slot_profile", []), dtype=np.float32) for bar in section_bars]
        similarity_matrix = np.eye(section_bar_count, dtype=np.float32)
        for left_index in range(section_bar_count):
            for right_index in range(left_index + 1, section_bar_count):
                similarity = self._slot_pattern_similarity(profiles[left_index], profiles[right_index])
                similarity_matrix[left_index, right_index] = similarity
                similarity_matrix[right_index, left_index] = similarity

        mean_support = np.mean(similarity_matrix, axis=1)
        prototype_local_indices = [int(np.argmax(mean_support))]
        while len(prototype_local_indices) < prototype_count and len(prototype_local_indices) < section_bar_count:
            best_candidate = -1
            best_distance = -1.0
            for local_index in range(section_bar_count):
                if local_index in prototype_local_indices:
                    continue
                min_similarity = max(float(similarity_matrix[local_index, proto_index]) for proto_index in prototype_local_indices)
                distance = 1.0 - min_similarity
                if distance > best_distance:
                    best_distance = distance
                    best_candidate = int(local_index)
            if best_candidate < 0:
                break
            prototype_local_indices.append(int(best_candidate))

        prototypes = []
        assignments = []
        assignment_counts = {}
        for local_index in range(section_bar_count):
            best_proto = prototype_local_indices[0]
            best_similarity = -1.0
            for proto_local_index in prototype_local_indices:
                similarity = float(similarity_matrix[local_index, proto_local_index])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_proto = int(proto_local_index)
            prototype_bar_index = int(start_index + best_proto)
            assignments.append(
                {
                    "bar_index": int(start_index + local_index),
                    "prototype_bar_index": prototype_bar_index,
                    "similarity": float(best_similarity),
                }
            )
            assignment_counts[prototype_bar_index] = int(assignment_counts.get(prototype_bar_index, 0) + 1)

        for proto_local_index in prototype_local_indices:
            absolute_index = int(start_index + proto_local_index)
            prototype_bar = bars[absolute_index]
            prototypes.append(
                {
                    "prototype_bar_index": absolute_index,
                    "active_slots": list(prototype_bar.get("preliminary_active_slots", [])),
                    "slot_profile": list(prototype_bar.get("slot_profile", [])),
                    "slot_count_template": [int(min(3, value)) for value in prototype_bar.get("slot_note_counts", [])],
                    "chosen_family": str(prototype_bar.get("chosen_family", "neutral")),
                    "section_state_divisor": int(prototype_bar.get("section_state_divisor", 0)),
                    "assigned_bars": int(assignment_counts.get(absolute_index, 0)),
                }
            )

        return {"prototypes": prototypes, "assignments": assignments}

    def _build_bar_pattern_cells(
        self,
        notes,
        onset_profile,
        hop_length,
        sr,
        bpm,
        first_beat_time=0,
        section_divisor_state=None,
        beat_family_state=None,
    ):
        if not notes or bpm <= 0:
            return {
                "bar_ms": 0.0,
                "slot_ms": 0.0,
                "origin_ms": 0.0,
                "bars": [],
                "sections": [],
            }

        beats_per_bar = max(1, int(self.config.BAR_PATTERN_BEATS))
        slots_per_beat = max(1, int(self.config.BAR_PATTERN_SLOTS_PER_BEAT))
        total_slots = beats_per_bar * slots_per_beat
        beat_duration_ms = 60000.0 / float(bpm)
        bar_ms = beat_duration_ms * float(beats_per_bar)
        slot_ms = beat_duration_ms / float(slots_per_beat)
        first_beat_ms = float(first_beat_time) * 1000.0
        bar_origin_ms = first_beat_ms
        while bar_origin_ms > 0.0:
            bar_origin_ms -= bar_ms

        profile = self._normalize_profile(onset_profile)
        last_note_ms = max(float(note.get("end_time", note.get("aligned_time", 0.0))) for note in notes)
        total_duration_ms = max(last_note_ms, float(profile.size - 1) * float(hop_length) / float(sr) * 1000.0)
        bars = []
        previous_active_slots = set()
        previous_family = "neutral"
        previous_state_divisor = 0
        bar_start_ms = bar_origin_ms

        while bar_start_ms <= total_duration_ms + bar_ms + 1e-3:
            bar_end_ms = bar_start_ms + bar_ms
            bar_mid_ms = bar_start_ms + bar_ms * 0.5
            family_window = self._resolve_beat_family_window(bar_mid_ms, beat_family_state)
            section_window = self._resolve_section_window(bar_mid_ms, section_divisor_state)
            chosen_family = str(family_window.get("chosen_family", "neutral")) if family_window else "neutral"
            chosen_state_divisor = int(section_window.get("chosen_state_divisor", 8)) if section_window else 8

            slot_onsets = np.zeros(total_slots, dtype=np.float32)
            slot_occupancy = np.zeros(total_slots, dtype=np.float32)
            slot_note_counts = np.zeros(total_slots, dtype=np.float32)
            for slot_index in range(total_slots):
                slot_time_ms = bar_start_ms + float(slot_index) * slot_ms
                slot_onsets[slot_index] = self._sample_profile_activity(profile, hop_length, sr, slot_time_ms)

            for note in notes:
                note_time_ms = float(note.get("aligned_time", 0.0))
                if note_time_ms < bar_start_ms or note_time_ms >= bar_end_ms:
                    continue
                slot_index = int(round((note_time_ms - bar_start_ms) / slot_ms))
                slot_index = max(0, min(slot_index, total_slots - 1))
                slot_occupancy[slot_index] += max(1.0, float(note.get("magnitude", 0.0)))
                slot_note_counts[slot_index] += 1.0

            max_occupancy = float(np.max(slot_occupancy)) if slot_occupancy.size > 0 else 0.0
            if max_occupancy > 1e-6:
                slot_occupancy /= max_occupancy

            raw_slot_scores = (
                slot_onsets * float(self.config.BAR_PATTERN_ONSET_WEIGHT)
                + slot_occupancy * float(self.config.BAR_PATTERN_OCCUPANCY_WEIGHT)
            )

            if previous_active_slots and chosen_family == previous_family and chosen_state_divisor == previous_state_divisor:
                for slot_index in previous_active_slots:
                    raw_slot_scores[slot_index] += float(self.config.BAR_PATTERN_INHERIT_BONUS)

            max_score = float(np.max(raw_slot_scores)) if raw_slot_scores.size > 0 else 0.0
            threshold = max_score * float(self.config.BAR_PATTERN_SLOT_THRESHOLD)
            preliminary_active_slots = {
                int(slot_index)
                for slot_index, score in enumerate(raw_slot_scores.tolist())
                if score >= threshold and max_score > 1e-6
            }

            if previous_active_slots and chosen_family == previous_family and chosen_state_divisor == previous_state_divisor:
                keep_threshold = max_score * float(self.config.BAR_PATTERN_NEIGHBOR_KEEP_THRESHOLD)
                for slot_index in previous_active_slots:
                    if raw_slot_scores[slot_index] >= keep_threshold:
                        preliminary_active_slots.add(int(slot_index))

            min_active_slots = max(1, int(self.config.BAR_PATTERN_MIN_ACTIVE_SLOTS))
            if len(preliminary_active_slots) < min_active_slots and max_score > 1e-6:
                ranked_slots = list(np.argsort(raw_slot_scores))[::-1]
                for slot_index in ranked_slots:
                    preliminary_active_slots.add(int(slot_index))
                    if len(preliminary_active_slots) >= min_active_slots:
                        break

            bars.append(
                {
                    "start_ms": float(bar_start_ms),
                    "end_ms": float(bar_end_ms),
                    "slot_ms": float(slot_ms),
                    "slot_scores": raw_slot_scores.tolist(),
                    "slot_profile": self._normalize_slot_vector(raw_slot_scores).tolist(),
                    "slot_onsets": slot_onsets.tolist(),
                    "slot_occupancy": slot_occupancy.tolist(),
                    "slot_note_counts": slot_note_counts.tolist(),
                    "preliminary_active_slots": sorted(preliminary_active_slots),
                    "active_slots": sorted(preliminary_active_slots),
                    "chosen_family": chosen_family,
                    "section_state_divisor": int(chosen_state_divisor),
                    "prototype_similarity": 0.0,
                    "prototype_bar_index": -1,
                    "prototype_slot_template": [0] * total_slots,
                    "bar_index": int(len(bars)),
                    "section_index": -1,
                }
            )
            previous_active_slots = set(preliminary_active_slots)
            previous_family = chosen_family
            previous_state_divisor = int(chosen_state_divisor)
            bar_start_ms += bar_ms

        sections, novelty, audio_novelty, combined_novelty = self._segment_bar_sections(bars)
        for section in sections:
            start_index = int(section.get("start_bar_index", 0))
            end_index = int(section.get("end_bar_index", start_index))
            motif_data = self._extract_section_bar_prototypes(bars, start_index, end_index)
            section["prototypes"] = motif_data.get("prototypes", [])
            section["assignments"] = motif_data.get("assignments", [])

            prototype_lookup = {
                int(item.get("prototype_bar_index", -1)): item
                for item in section["prototypes"]
            }
            assignment_lookup = {
                int(item.get("bar_index", -1)): item
                for item in section["assignments"]
            }

            for bar_index in range(start_index, end_index):
                bar = bars[bar_index]
                bar["section_index"] = int(section.get("section_index", -1))
                assignment = assignment_lookup.get(int(bar_index))
                if assignment is None:
                    continue
                prototype_bar_index = int(assignment.get("prototype_bar_index", -1))
                prototype = prototype_lookup.get(prototype_bar_index)
                raw_slot_scores = np.asarray(bar.get("slot_scores", []), dtype=np.float32)
                max_score = float(np.max(raw_slot_scores)) if raw_slot_scores.size > 0 else 0.0
                prototype_slots = list(prototype.get("active_slots", [])) if prototype else list(bar.get("preliminary_active_slots", []))
                initial_active_slots = set(int(value) for value in bar.get("preliminary_active_slots", []))
                final_slots = set()

                keep_ratio = float(self.config.BAR_PATTERN_PROTOTYPE_KEEP_RATIO)
                for slot_index in prototype_slots:
                    slot_index = int(slot_index)
                    if slot_index in initial_active_slots or (max_score > 1e-6 and raw_slot_scores[slot_index] >= max_score * keep_ratio):
                        final_slots.add(slot_index)

                if bool(self.config.BAR_PATTERN_PROTOTYPE_FORCE_COPY) and prototype_slots and len(final_slots) < max(1, min_active_slots):
                    ranked_proto_slots = sorted(
                        set(int(value) for value in prototype_slots),
                        key=lambda slot: float(raw_slot_scores[slot]),
                        reverse=True,
                    )
                    for slot_index in ranked_proto_slots:
                        final_slots.add(int(slot_index))
                        if len(final_slots) >= max(1, min_active_slots):
                            break

                extra_threshold = max_score * float(self.config.BAR_PATTERN_PROTOTYPE_EXTRA_RATIO)
                max_extra_slots = max(0, int(self.config.BAR_PATTERN_PROTOTYPE_MAX_EXTRA_SLOTS))
                extra_candidates = [
                    int(slot_index)
                    for slot_index, score in enumerate(raw_slot_scores.tolist())
                    if int(slot_index) not in final_slots and score >= extra_threshold and max_score > 1e-6
                ]
                extra_candidates.sort(key=lambda slot: float(raw_slot_scores[slot]), reverse=True)
                for slot_index in extra_candidates[:max_extra_slots]:
                    final_slots.add(int(slot_index))

                if len(final_slots) < min_active_slots and max_score > 1e-6:
                    ranked_slots = list(np.argsort(raw_slot_scores))[::-1]
                    for slot_index in ranked_slots:
                        final_slots.add(int(slot_index))
                        if len(final_slots) >= min_active_slots:
                            break

                bar["active_slots"] = sorted(final_slots)
                bar["prototype_bar_index"] = int(prototype_bar_index)
                bar["prototype_similarity"] = float(assignment.get("similarity", 0.0))
                bar["prototype_slot_template"] = list(prototype.get("slot_count_template", [])) if prototype else [0] * total_slots

        return {
            "bar_ms": float(bar_ms),
            "slot_ms": float(slot_ms),
            "origin_ms": float(bar_origin_ms),
            "bars": bars,
            "sections": sections,
            "novelty": novelty,
            "audio_novelty": audio_novelty,
            "combined_novelty": combined_novelty,
        }

    def _resolve_bar_pattern_cell(self, time_ms, bar_pattern_cells):
        if not bar_pattern_cells:
            return None
        bars = bar_pattern_cells.get("bars", [])
        if not bars:
            return None
        bar_ms = float(bar_pattern_cells.get("bar_ms", 0.0))
        origin_ms = float(bar_pattern_cells.get("origin_ms", 0.0))
        if bar_ms <= 0:
            return None
        bar_index = int((float(time_ms) - origin_ms) // bar_ms)
        bar_index = max(0, min(bar_index, len(bars) - 1))
        return bars[bar_index]


    def apply_bar_pattern_cell_filter(
        self,
        notes,
        onset_profile,
        hop_length,
        sr,
        bpm,
        first_beat_time=0,
        section_divisor_state=None,
        beat_family_state=None,
    ):
        print("应用 bar-level pattern cell...")

        if not notes or bpm <= 0:
            return []

        bar_pattern_cells = self._build_bar_pattern_cells(
            notes,
            onset_profile,
            hop_length,
            sr,
            bpm,
            first_beat_time,
            section_divisor_state,
            beat_family_state,
        )
        if not bar_pattern_cells.get("bars", []):
            return [dict(note) for note in notes]

        bars = bar_pattern_cells.get("bars", [])
        notes_by_bar = {}
        passthrough_notes = []

        for source_note in notes:
            note = dict(source_note)
            note_time_ms = float(note.get("aligned_time", 0.0))
            bar_cell = self._resolve_bar_pattern_cell(note_time_ms, bar_pattern_cells)
            if bar_cell is None:
                passthrough_notes.append(note)
                continue
            bar_index = int(bar_cell.get("bar_index", -1))
            if bar_index < 0 or bar_index >= len(bars):
                passthrough_notes.append(note)
                continue
            notes_by_bar.setdefault(bar_index, []).append(note)

        adjusted_notes = list(passthrough_notes)
        remapped_count = 0

        for bar_index in sorted(notes_by_bar.keys()):
            bar_cell = bars[bar_index]
            chosen_family = str(bar_cell.get("chosen_family", "neutral"))
            active_slots = [int(value) for value in bar_cell.get("active_slots", [])]
            slot_ms = float(bar_cell.get("slot_ms", bar_pattern_cells.get("slot_ms", 0.0)))
            if chosen_family != "binary" or not active_slots or slot_ms <= 0.0:
                adjusted_notes.extend(notes_by_bar[bar_index])
                continue

            bar_start_ms = float(bar_cell.get("start_ms", 0.0))
            section_state_divisor = int(bar_cell.get("section_state_divisor", 0))
            prototype_slot_template = list(bar_cell.get("prototype_slot_template", []))
            if not prototype_slot_template:
                prototype_slot_template = [0] * max(active_slots + [0, 0])
            template_capacity = {
                int(slot_index): max(
                    1,
                    int(prototype_slot_template[slot_index]) + int(self.config.BAR_PATTERN_PROTOTYPE_TEMPLATE_SLACK)
                    if slot_index < len(prototype_slot_template)
                    else 1,
                )
                for slot_index in active_slots
            }
            assigned_counts = {int(slot_index): 0 for slot_index in active_slots}

            for source_note in sorted(notes_by_bar[bar_index], key=lambda item: float(item.get("aligned_time", 0.0))):
                note = dict(source_note)
                note_family = self._classify_beat_family(int(note.get("snap_divisor", 0)))
                remap_enabled = False
                note_divisor = int(note.get("snap_divisor", 0))
                if note_family == "triplet" and bool(self.config.BAR_PATTERN_ENABLE_TRIPLET_REMAP):
                    remap_enabled = True
                elif (
                    bool(self.config.BAR_PATTERN_ENABLE_FINE_BINARY_REMAP)
                    and note_divisor > max(2, section_state_divisor)
                ):
                    remap_enabled = True

                if not remap_enabled:
                    adjusted_notes.append(note)
                    continue

                note_time_ms = float(note.get("aligned_time", 0.0))
                current_slot_index = int(round((note_time_ms - bar_start_ms) / slot_ms))
                current_slot_index = max(0, min(current_slot_index, max(active_slots + [0])))
                if current_slot_index in active_slots:
                    adjusted_notes.append(note)
                    assigned_counts[int(current_slot_index)] = int(assigned_counts.get(int(current_slot_index), 0) + 1)
                    continue

                slot_candidates = []
                best_delta_ms = None
                for slot_index in active_slots:
                    slot_time_ms = bar_start_ms + float(slot_index) * slot_ms
                    delta_ms = abs(slot_time_ms - note_time_ms)
                    slot_candidates.append((int(slot_index), float(slot_time_ms), float(delta_ms)))
                    if best_delta_ms is None or delta_ms < best_delta_ms:
                        best_delta_ms = float(delta_ms)

                tie_margin_ms = float(slot_ms) * float(self.config.BAR_PATTERN_PROTOTYPE_TEMPLATE_TIE_MARGIN_RATIO)
                candidate_slots = [
                    item
                    for item in slot_candidates
                    if item[2] <= float(best_delta_ms or 0.0) + tie_margin_ms
                ] or slot_candidates

                best_slot = candidate_slots[0][0]
                best_time_ms = candidate_slots[0][1]
                best_cost = None
                for slot_index, slot_time_ms, delta_ms in candidate_slots:
                    overflow = max(
                        0,
                        assigned_counts.get(int(slot_index), 0)
                        - template_capacity.get(int(slot_index), 1)
                        + 1,
                    )
                    overflow_penalty = (
                        float(slot_ms)
                        * float(self.config.BAR_PATTERN_PROTOTYPE_TEMPLATE_PENALTY)
                        * float(overflow)
                    )
                    cost = float(delta_ms + overflow_penalty)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_slot = int(slot_index)
                        best_time_ms = float(slot_time_ms)

                nearest_delta_ms = abs(best_time_ms - note_time_ms)

                tolerance_ms = slot_ms * float(self.config.BAR_PATTERN_SNAP_TOLERANCE_RATIO)
                if nearest_delta_ms > tolerance_ms:
                    adjusted_notes.append(note)
                    continue

                if abs(best_time_ms - note_time_ms) <= 0.5:
                    adjusted_notes.append(note)
                    assigned_counts[int(best_slot)] = int(assigned_counts.get(int(best_slot), 0) + 1)
                    continue

                time_shift_ms = best_time_ms - note_time_ms
                note["aligned_time"] = float(best_time_ms)
                if float(note.get("duration", 0.0)) > 0.0:
                    note["end_time"] = float(note.get("end_time", note_time_ms)) + time_shift_ms
                else:
                    note["end_time"] = note["aligned_time"] + float(note.get("duration", 0.0))
                note["snap_divisor"] = 1 if (best_slot % 2 == 0) else 2
                adjusted_notes.append(note)
                assigned_counts[int(best_slot)] = int(assigned_counts.get(int(best_slot), 0) + 1)
                remapped_count += 1

        adjusted_notes.sort(key=lambda item: float(item.get("aligned_time", 0.0)))
        print(f"bar pattern cell: 重映射了 {remapped_count}/{len(notes)} 个音符")
        return adjusted_notes

    def _build_section_divisor_weight_map(self, time_ms, divisor_weights, section_divisor_state, beat_family_state=None):
        base_weights = dict(divisor_weights or {})
        if not section_divisor_state:
            resolved_weights = dict(base_weights)
        else:
            window = self._resolve_section_window(time_ms, section_divisor_state)
            if window is None:
                resolved_weights = dict(base_weights)
            else:
                chosen_state = int(window.get("chosen_state_divisor", 0))
                normalized_scores = window.get("normalized_local_scores", {})
                anchor_divisors = set(int(value) for value in window.get("anchor_divisors", []))
                strongest_anchor = min(anchor_divisors) if anchor_divisors else int(chosen_state)
                strongest_anchor_score = float(normalized_scores.get(int(strongest_anchor), 0.25))
                resolved_weights = {}

                for divisor in self.config.BEAT_DIVISORS:
                    base_weight = float(base_weights.get(int(divisor), self.config.GRID_WEIGHT_PRIORS.get(int(divisor), 0.2)))
                    tier = self._map_divisor_to_section_tier(int(divisor))
                    local_scale = float(normalized_scores.get(int(tier), 0.25))
                    if tier > chosen_state:
                        multiplier = float(self.config.SECTION_STATE_SUPPRESS_FINESCALE_MULTIPLIER)
                    elif tier == chosen_state:
                        multiplier = 0.65 + local_scale * 0.55
                    else:
                        multiplier = 0.45 + local_scale * 0.35
                    if int(tier) in anchor_divisors:
                        multiplier *= float(self.config.SECTION_STATE_ANCHOR_BOOST)
                    if int(tier) == int(strongest_anchor):
                        multiplier *= float(self.config.SECTION_STATE_COARSE_SKELETON_BOOST)
                    elif int(tier) > int(strongest_anchor):
                        tier_score = float(normalized_scores.get(int(tier), 0.05))
                        if tier_score < strongest_anchor_score + float(self.config.SECTION_STATE_COARSE_PROTECTION_MARGIN):
                            multiplier *= float(self.config.SECTION_STATE_FINE_COMPETITION_PENALTY)
                    resolved_weights[int(divisor)] = base_weight * max(0.05, float(multiplier))

        if beat_family_state:
            global_family = str(beat_family_state.get("global_family", "neutral"))
            family_window = self._resolve_beat_family_window(time_ms, beat_family_state)
            chosen_family = str(family_window.get("chosen_family", "neutral")) if family_window else "neutral"
            for divisor in list(resolved_weights.keys()):
                family = self._classify_beat_family(divisor)
                multiplier = 1.0
                if family != "neutral" and global_family != "neutral" and family != global_family:
                    multiplier *= float(self.config.BEAT_FAMILY_GLOBAL_SUPPRESS_MULTIPLIER)
                if family != "neutral" and chosen_family != "neutral":
                    if family == chosen_family:
                        multiplier *= float(self.config.BEAT_FAMILY_WINDOW_MATCH_BOOST)
                    else:
                        multiplier *= float(self.config.BEAT_FAMILY_WINDOW_SUPPRESS_MULTIPLIER)
                resolved_weights[int(divisor)] = max(0.01, float(resolved_weights[int(divisor)]) * float(multiplier))

        return resolved_weights

    def _snap_time_to_grid(
        self, time_ms, beat_duration_ms, first_beat_ms, divisors, divisor_weights=None
    ):
        if beat_duration_ms <= 0 or not divisors:
            return float(time_ms), 0, 0.0

        raw_beat_pos = (float(time_ms) - float(first_beat_ms)) / float(beat_duration_ms)
        candidates = []

        for divisor in divisors:
            snapped_pos = round(raw_beat_pos * divisor) / divisor
            error_ms = abs(raw_beat_pos - snapped_pos) * beat_duration_ms
            weight = 0.0
            if divisor_weights is not None:
                weight = float(divisor_weights.get(int(divisor), 0.0))
            if weight <= 0.0:
                weight = float(self.config.GRID_WEIGHT_PRIORS.get(int(divisor), 0.2))
            candidates.append((float(error_ms), int(divisor), float(snapped_pos), weight))

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
        preferred_candidates.sort(
            key=lambda candidate: (-candidate[3], candidate[1], candidate[0])
        )
        chosen_error, best_divisor, best_snap, _ = preferred_candidates[0]

        snapped_time = float(first_beat_ms + best_snap * beat_duration_ms)
        return snapped_time, best_divisor, float(chosen_error)

    def align_to_beat_grid(
        self,
        note_events,
        bpm,
        first_beat_time=0,
        divisor_weights=None,
        section_divisor_state=None,
        beat_family_state=None,
    ):
        print("多级节拍网格对齐...")
        aligned_notes = []
        beat_duration_ms = 60000 / bpm if bpm else 0
        first_beat_ms = first_beat_time * 1000
        valid_count = 0

        for source_note in note_events:
            note = dict(source_note)
            start_time = float(note["start_time"])
            section_weight_map = self._build_section_divisor_weight_map(
                start_time,
                divisor_weights,
                section_divisor_state,
                beat_family_state,
            )
            (
                snapped_time,
                snapped_divisor,
                min_error,
            ) = self._snap_time_to_grid(
                start_time,
                beat_duration_ms,
                first_beat_ms,
                self.config.BEAT_DIVISORS,
                section_weight_map,
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

    def apply_timing_grid_filter(
        self,
        aligned_notes,
        bpm,
        first_beat_time=0,
        divisor_weights=None,
        section_divisor_state=None,
        beat_family_state=None,
    ):
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
            start_weight_map = self._build_section_divisor_weight_map(
                original_start,
                divisor_weights,
                section_divisor_state,
                beat_family_state,
            )
            (
                snapped_start,
                snapped_divisor,
                _,
            ) = self._snap_time_to_grid(
                original_start,
                beat_duration_ms,
                first_beat_ms,
                self.config.BEAT_DIVISORS,
                start_weight_map,
            )
            if abs(snapped_start - original_start) > 0.5:
                adjusted_count += 1

            note["aligned_time"] = snapped_start
            note["snap_divisor"] = snapped_divisor

            original_end = float(note.get("end_time", original_start + note.get("duration", 0.0)))
            if original_end > original_start:
                end_weight_map = self._build_section_divisor_weight_map(
                    original_end,
                    divisor_weights,
                    section_divisor_state,
                    beat_family_state,
                )
                snapped_end, _, _ = self._snap_time_to_grid(
                    original_end,
                    beat_duration_ms,
                    first_beat_ms,
                    hold_divisors,
                    end_weight_map,
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
        self, aligned_notes, energy_profile, hop_length, sr, beat_duration_ms, divisor_weights=None
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
                    key=lambda x: (
                        x.get("magnitude", 0) * 0.7
                        + float((divisor_weights or {}).get(int(x.get("snap_divisor", 0)), 0.0)) * 0.45,
                        float((divisor_weights or {}).get(int(x.get("snap_divisor", 0)), 0.0)),
                        x["duration"],
                    ),
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

        onset_profiles = audio_data.get("onset_profiles", {})
        combined_onset_profile = onset_profiles.get("combined", audio_data["energy_profile"])
        bpm_info = self.detect_bpm(audio_data["audio"], audio_data["sample_rate"], combined_onset_profile)
        beat_path = self.build_dynamic_beat_path(
            combined_onset_profile,
            audio_data["hop_length"],
            audio_data["sample_rate"],
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
        )
        if beat_path.get("beat_times_ms"):
            bpm_info["first_beat_time"] = float(beat_path["beat_times_ms"][0]) / 1000.0
        beat_duration_ms = 60000 / bpm_info["bpm"] if bpm_info["bpm"] else 0
        grid_weight_metrics, grid_divisor_weights = self.build_divisor_weight_metrics(
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            combined_onset_profile,
        )
        transient_layer_metrics, transient_layer_weights = self.build_transient_layer_metrics(
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            onset_profiles,
        )
        combined_divisor_weights = dict(grid_divisor_weights)
        for divisor, transient_weight in transient_layer_weights.items():
            base_weight = float(combined_divisor_weights.get(divisor, 0.0))
            if base_weight <= 0.0:
                combined_divisor_weights[divisor] = transient_weight
            else:
                combined_divisor_weights[divisor] = base_weight * 0.55 + transient_weight * 0.45
        print(
            "节拍层权重: "
            + ", ".join(
                [
                    f"1/{divisor}={combined_divisor_weights.get(divisor, 0.0):.2f}"
                    for divisor in [1, 2, 4, 8]
                    if divisor in combined_divisor_weights
                ]
            )
        )
        section_divisor_state = self.build_section_divisor_state(
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            onset_profiles,
            combined_divisor_weights,
        )
        beat_family_state = self.build_beat_family_state(
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            onset_profiles,
            combined_divisor_weights,
        )
        stable_segments = section_divisor_state.get("stable_segments", [])
        if stable_segments:
            preview_segments = ", ".join(
                [
                    f"{int(segment['start_ms'])}-{int(segment['end_ms'])}ms=1/{int(segment['state_divisor'])}"
                    for segment in stable_segments[:4]
                ]
            )
            print(f"窗口分音层状态: {preview_segments}")
        family_segments = beat_family_state.get("stable_segments", [])
        if family_segments:
            preview_families = ", ".join(
                [
                    f"{int(segment['start_ms'])}-{int(segment['end_ms'])}ms={segment['family']}"
                    for segment in family_segments[:4]
                ]
            )
            print(
                f"节拍家族状态: global={beat_family_state.get('global_family', 'neutral')} | {preview_families}"
            )

        aligned_notes = self.align_to_beat_grid(
            note_events,
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            combined_divisor_weights,
            section_divisor_state,
            beat_family_state,
        )
        timing_filtered_notes = self.apply_timing_grid_filter(
            aligned_notes,
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            combined_divisor_weights,
            section_divisor_state,
            beat_family_state,
        )
        bar_pattern_notes = self.apply_bar_pattern_cell_filter(
            timing_filtered_notes,
            combined_onset_profile,
            audio_data["hop_length"],
            audio_data["sample_rate"],
            bpm_info["bpm"],
            bpm_info["first_beat_time"],
            section_divisor_state,
            beat_family_state,
        )
        density_filtered_notes = self.apply_dynamic_density_filter(
            bar_pattern_notes,
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
            beat_duration_ms,
            combined_divisor_weights,
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
            "beat_path": beat_path,
            "aligned_notes": aligned_notes,
            "timing_filtered_notes": timing_filtered_notes,
            "bar_pattern_notes": bar_pattern_notes,
            "density_filtered_notes": density_filtered_notes,
            "silence_filtered_notes": silence_filtered_notes,
            "mapped_notes": mapped_notes,
            "balanced_notes": balanced_notes,
            "controlled_notes": final_notes,
            "grid_weight_metrics": grid_weight_metrics,
            "grid_divisor_weights": combined_divisor_weights,
            "transient_layer_metrics": transient_layer_metrics,
            "transient_layer_weights": transient_layer_weights,
            "section_divisor_state": section_divisor_state,
            "beat_family_state": beat_family_state,
            "config": {
                "columns": self.config.DEFAULT_COLUMNS,
                "sample_rate": audio_data["sample_rate"],
                "hop_length": audio_data["hop_length"],
            },
        }

