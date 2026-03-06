import librosa
import numpy as np


class FrequencyAnalyzer:
    def __init__(self, bpm=None):
        self.bpm = bpm
        self.target_frequencies = []

    def analyze_bpm(self, audio_path):
        """
        使用 librosa 分析音频的 BPM。
        """
        y, sr = librosa.load(audio_path, sr=None)
        self.bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return self.bpm

    def calculate_frequencies(self):
        """
        根据 BPM 计算目标频率列表。
        """
        base_frequency = self.bpm / 60
        divisions = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
        self.target_frequencies = [base_frequency * div for div in divisions]
        return self.target_frequencies

    def analyze_signals(self, signals, fs=256, window_size=512, hop_size=256):
        """
        对按键信号进行频率分析。
        """
        amplitude_matrices = []
        time_axes = None
        for signal in signals:
            n_samples = len(signal)
            n_frames = (n_samples - window_size) // hop_size + 1
            time_axis = np.arange(n_frames) * hop_size / fs
            amplitude_matrix = np.zeros((len(self.target_frequencies), n_frames))

            for i, freq in enumerate(self.target_frequencies):
                omega = 2 * np.pi * freq / fs
                for frame_idx in range(n_frames):
                    start = frame_idx * hop_size
                    end = start + window_size
                    if end > n_samples:
                        break
                    windowed_signal = signal[start:end] * np.hanning(window_size)
                    amplitude_matrix[i, frame_idx] = np.abs(
                        np.sum(
                            windowed_signal
                            * np.exp(-1j * omega * np.arange(window_size))
                        )
                    )

            amplitude_matrices.append(amplitude_matrix)
            time_axes = time_axis
        return amplitude_matrices, time_axes
