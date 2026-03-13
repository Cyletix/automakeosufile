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

        _, _, stft_matrix = signal.stft(
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

        return mel_spec, log_mel, mel_normalized

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
        mel_spec, log_mel, mel_normalized = self.compute_mel_spectrogram(y, sr)
        binary_matrix = self.adaptive_binarization(mel_normalized)
        energy_profile = self.calculate_energy(y, self.config.HOP_LENGTH)
        note_events = self.extract_note_events(
            binary_matrix, log_mel, sr, self.config.HOP_LENGTH
        )

        return {
            "audio": y,
            "sample_rate": sr,
            "mel_spectrogram": mel_spec,
            "log_mel": log_mel,
            "binary_matrix": binary_matrix,
            "note_events": note_events,
            "energy_profile": energy_profile,
            "hop_length": self.config.HOP_LENGTH,
        }
