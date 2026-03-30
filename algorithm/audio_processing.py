# -*- coding: utf-8 -*-
"""
Audio processing: load audio, compute Mel spectrogram, binarize, and extract
note events without relying on librosa-heavy runtime paths.
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
import soundfile as sf
from scipy import signal

from .config import Config


class AudioProcessor:
    def __init__(self, config=None):
        self.config = config or Config()

    def load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        temp_dir = None

        try:
            print(f"加载音频: {audio_path}")
            y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except RuntimeError:
            if not audio_path.lower().endswith(".mp3"):
                raise

            temp_dir = tempfile.mkdtemp(prefix="automakeosu_audio_")
            wav_path = os.path.join(
                temp_dir,
                os.path.splitext(os.path.basename(audio_path))[0] + ".wav",
            )
            print(f"转换MP3到临时WAV: {audio_path}")
            self._convert_audio(audio_path, wav_path)
            print(f"加载音频: {wav_path}")
            y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if getattr(y, "ndim", 1) > 1 and self.config.MONO:
            y = np.mean(y, axis=1)

        if self.config.DURATION is not None:
            y = y[: int(sr * self.config.DURATION)]

        if sr != self.config.SAMPLE_RATE:
            y = signal.resample_poly(y, self.config.SAMPLE_RATE, sr)
            sr = self.config.SAMPLE_RATE

        return y.astype(np.float32), sr

    def _convert_audio(self, source_path, wav_path):
        audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        sf.write(wav_path, audio, sample_rate)

    def compute_mel_spectrogram(self, y, sr):
        print("计算Mel频谱图...")

        frequencies_hz, _, stft_matrix = signal.stft(
            y,
            fs=sr,
            nperseg=self.config.N_FFT,
            noverlap=self.config.N_FFT - self.config.HOP_LENGTH,
            padded=False,
            boundary=None,
        )
        power_spectrogram = np.abs(stft_matrix) ** 2

        mel_filter = self._build_mel_filter(sr)
        mel_spec = np.dot(mel_filter, power_spectrogram)
        mel_spec = np.maximum(mel_spec, 1e-10)

        log_mel = 10.0 * np.log10(mel_spec)
        log_mel -= np.max(log_mel)
        mel_normalized = self._normalize_to_uint8(log_mel)

        return mel_spec, log_mel, mel_normalized, power_spectrogram, frequencies_hz

    def _build_mel_filter(self, sr):
        n_fft = self.config.N_FFT
        n_mels = self.config.N_MELS
        fmin = self.config.FMIN
        fmax = min(self.config.FMAX, sr / 2)

        mel_min = self._hz_to_mel(fmin)
        mel_max = self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bins = np.clip(bins, 0, n_fft // 2)

        filter_bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

        for index in range(1, n_mels + 1):
            left = bins[index - 1]
            center = bins[index]
            right = bins[index + 1]

            if center == left:
                center += 1
            if right == center:
                right += 1

            for frequency_bin in range(left, center):
                filter_bank[index - 1, frequency_bin] = (frequency_bin - left) / (
                    center - left
                )
            for frequency_bin in range(center, right):
                filter_bank[index - 1, frequency_bin] = (right - frequency_bin) / (
                    right - center
                )

        return filter_bank

    def adaptive_binarization(self, mel_normalized):
        print("自适应二值化...")
        binary = cv2.adaptiveThreshold(
            mel_normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            self.config.ADAPTIVE_THRESHOLD_C,
        )

        kernel = np.ones(
            (self.config.MORPH_KERNEL_SIZE, self.config.MORPH_KERNEL_SIZE), np.uint8
        )
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return (binary_cleaned > 0).astype(np.float32)

    def calculate_energy(self, y, hop_length):
        frame_length = self.config.N_FFT
        if len(y) < frame_length:
            return np.array([0.0], dtype=np.float32)

        frame_count = 1 + (len(y) - frame_length) // hop_length
        rms_values = np.empty(frame_count, dtype=np.float32)

        for index in range(frame_count):
            start = index * hop_length
            frame = y[start : start + frame_length]
            rms_values[index] = np.sqrt(np.mean(np.square(frame)))

        window_size = self.config.ENERGY_SMOOTHING_WINDOW
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            rms_values = np.convolve(rms_values, kernel, mode="same")

        rms_min = float(np.min(rms_values))
        rms_max = float(np.max(rms_values))
        if rms_max > rms_min:
            return (rms_values - rms_min) / (rms_max - rms_min)
        return np.zeros_like(rms_values)

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

    def _positive_diff_profile(self, values):
        profile = np.asarray(values, dtype=np.float32)
        if profile.size == 0:
            return np.zeros(1, dtype=np.float32)
        diff_profile = np.maximum(0.0, np.diff(profile, prepend=profile[0]))
        window_size = max(1, int(self.config.ONSET_SMOOTHING_WINDOW))
        if window_size > 1:
            kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
            diff_profile = np.convolve(diff_profile, kernel, mode="same")
        return self._normalize_profile(diff_profile)

    def calculate_onset_profiles(self, mel_spectrogram):
        mel = np.asarray(mel_spectrogram, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[1] == 0:
            zero_profile = np.zeros(1, dtype=np.float32)
            return {
                "low": zero_profile,
                "mid": zero_profile,
                "high": zero_profile,
                "combined": zero_profile,
            }

        band_count = mel.shape[0]

        def band_slice(ratio_range):
            start_ratio, end_ratio = ratio_range
            start_index = max(
                0, min(band_count - 1, int(round(band_count * start_ratio)))
            )
            end_index = max(
                start_index + 1, min(band_count, int(round(band_count * end_ratio)))
            )
            return mel[start_index:end_index, :]

        low_band = band_slice(self.config.ONSET_LOW_BAND_RATIO)
        mid_band = band_slice(self.config.ONSET_MID_BAND_RATIO)
        high_band = band_slice(self.config.ONSET_HIGH_BAND_RATIO)

        low_profile = self._positive_diff_profile(np.mean(low_band, axis=0))
        mid_profile = self._positive_diff_profile(np.mean(mid_band, axis=0))
        high_profile = self._positive_diff_profile(np.mean(high_band, axis=0))

        low_weight, mid_weight, high_weight = self.config.ONSET_COMBINED_WEIGHTS
        combined_profile = self._normalize_profile(
            low_profile * float(low_weight)
            + mid_profile * float(mid_weight)
            + high_profile * float(high_weight)
        )

        return {
            "low": low_profile,
            "mid": mid_profile,
            "high": high_profile,
            "combined": combined_profile,
        }

    @staticmethod
    def _hz_to_midi(frequency_hz):
        safe_frequency = np.maximum(np.asarray(frequency_hz, dtype=np.float32), 1e-6)
        return 69.0 + 12.0 * np.log2(safe_frequency / 440.0)

    def compute_pitch_salience_roll(self, power_spectrogram, frequencies_hz):
        print("计算Pitch Salience Roll...")
        midi_min = int(self.config.PITCH_MIDI_MIN)
        midi_max = int(self.config.PITCH_MIDI_MAX)
        midi_axis = np.arange(midi_min, midi_max + 1, dtype=np.float32)
        if power_spectrogram.ndim != 2 or power_spectrogram.shape[1] == 0:
            empty_roll = np.zeros((midi_axis.size, 1), dtype=np.float32)
            return (
                empty_roll,
                np.zeros_like(empty_roll, dtype=np.uint8),
                midi_axis,
                np.zeros(1, dtype=np.float32),
            )

        roll = np.zeros((midi_axis.size, power_spectrogram.shape[1]), dtype=np.float32)
        valid_mask = frequencies_hz > max(20.0, float(self.config.FMIN))
        valid_frequencies = frequencies_hz[valid_mask]
        valid_power = power_spectrogram[valid_mask, :]
        midi_values = self._hz_to_midi(valid_frequencies)
        midi_indices = np.round(midi_values).astype(int) - midi_min
        valid_index_mask = (midi_indices >= 0) & (midi_indices < midi_axis.size)
        midi_indices = midi_indices[valid_index_mask]
        valid_power = valid_power[valid_index_mask, :]

        for row_index, midi_index in enumerate(midi_indices.tolist()):
            roll[int(midi_index), :] += valid_power[row_index, :]

        if roll.size > 0:
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            roll = signal.convolve2d(roll, kernel, mode="same", boundary="symm").astype(
                np.float32
            )
            roll = np.log1p(np.maximum(roll, 0.0))
            roll_max = float(np.max(roll))
            if roll_max > 1e-6:
                roll /= roll_max

        pitch_normalized = np.clip(np.round(roll * 255.0), 0, 255).astype(np.uint8)
        pitch_onset_profile = self._positive_diff_profile(np.max(roll, axis=0))
        return roll, pitch_normalized, midi_axis, pitch_onset_profile

    def extract_note_events(self, binary_matrix, log_mel, sr, hop_length):
        print("提取音符事件(含强度检测)...")
        note_events = []
        n_freq_bins, _ = binary_matrix.shape
        time_per_frame = hop_length / sr

        for freq_bin in range(n_freq_bins):
            time_series = binary_matrix[freq_bin, :]
            changes = np.diff(np.concatenate(([0], time_series, [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            for start_frame, end_frame in zip(starts, ends):
                duration_frames = end_frame - start_frame
                duration_ms = duration_frames * time_per_frame * 1000

                if not (
                    self.config.MIN_NOTE_DURATION_MS
                    <= duration_ms
                    <= self.config.MAX_NOTE_DURATION_MS
                ):
                    continue

                start_time = start_frame * time_per_frame * 1000
                end_time = end_frame * time_per_frame * 1000
                magnitude = float(np.mean(log_mel[freq_bin, start_frame:end_frame]))

                note_events.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration_ms,
                        "frequency_bin": freq_bin,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "magnitude": magnitude,
                    }
                )

        note_events.sort(key=lambda item: item["start_time"])
        print(f"提取到 {len(note_events)} 个音符事件")
        return note_events

    def extract_pitch_note_events(
        self, binary_matrix, pitch_roll, midi_axis, sr, hop_length
    ):
        print("提取Pitch Symbolic候选...")
        note_events = []
        n_pitch_bins, _ = binary_matrix.shape
        time_per_frame = hop_length / sr

        for pitch_bin in range(n_pitch_bins):
            time_series = binary_matrix[pitch_bin, :]
            changes = np.diff(np.concatenate(([0], time_series, [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            for start_frame, end_frame in zip(starts, ends):
                duration_frames = end_frame - start_frame
                duration_ms = duration_frames * time_per_frame * 1000
                if not (
                    self.config.MIN_NOTE_DURATION_MS
                    <= duration_ms
                    <= self.config.MAX_NOTE_DURATION_MS
                ):
                    continue

                start_time = start_frame * time_per_frame * 1000
                end_time = end_frame * time_per_frame * 1000
                magnitude = float(np.mean(pitch_roll[pitch_bin, start_frame:end_frame]))
                pitch_midi = float(midi_axis[pitch_bin])
                pitch_class = int(round(pitch_midi)) % 12
                note_events.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration_ms,
                        "frequency_bin": int(pitch_bin),
                        "pitch_midi": pitch_midi,
                        "pitch_class": pitch_class,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "magnitude": magnitude,
                        "source": "pitch_salience",
                    }
                )

        note_events.sort(key=lambda item: item["start_time"])
        print(f"提取到 {len(note_events)} 个Pitch候选")
        return note_events

    def _normalize_to_uint8(self, data):
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if data_max > data_min:
            normalized = 255 * (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        return normalized.astype(np.uint8)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def process_audio(self, audio_path):
        y, sr = self.load_audio(audio_path)
        mel_spec, log_mel, mel_normalized, power_spectrogram, frequencies_hz = (
            self.compute_mel_spectrogram(y, sr)
        )
        binary_matrix = self.adaptive_binarization(mel_normalized)
        pitch_roll, pitch_normalized, pitch_midi_axis, pitch_onset_profile = (
            self.compute_pitch_salience_roll(
                power_spectrogram,
                frequencies_hz,
            )
        )
        pitch_binary_matrix = self.adaptive_binarization(pitch_normalized)
        energy_profile = self.calculate_energy(y, self.config.HOP_LENGTH)
        onset_profiles = self.calculate_onset_profiles(mel_spec)
        onset_profiles["pitch"] = pitch_onset_profile
        pitch_blend = float(self.config.PITCH_ONSET_BLEND_WEIGHT)
        onset_profiles["combined"] = self._normalize_profile(
            onset_profiles["combined"] * max(0.0, 1.0 - pitch_blend)
            + pitch_onset_profile * max(0.0, pitch_blend)
        )
        mel_note_events = self.extract_note_events(
            binary_matrix, log_mel, sr, self.config.HOP_LENGTH
        )
        pitch_note_events = self.extract_pitch_note_events(
            pitch_binary_matrix,
            pitch_roll,
            pitch_midi_axis,
            sr,
            self.config.HOP_LENGTH,
        )
        note_events = (
            pitch_note_events
            if bool(self.config.ENABLE_PITCH_SALIENCE_SOURCE) and pitch_note_events
            else mel_note_events
        )

        return {
            "audio": y,
            "sample_rate": sr,
            "mel_spectrogram": mel_spec,
            "log_mel": log_mel,
            "binary_matrix": binary_matrix,
            "pitch_salience_roll": pitch_roll,
            "pitch_binary_matrix": pitch_binary_matrix,
            "pitch_midi_axis": pitch_midi_axis,
            "mel_note_events": mel_note_events,
            "pitch_note_events": pitch_note_events,
            "note_events": note_events,
            "energy_profile": energy_profile,
            "onset_profiles": onset_profiles,
            "hop_length": self.config.HOP_LENGTH,
        }
