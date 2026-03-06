"""
音频处理模块 - 加载音频、计算Mel频谱、自适应二值化
"""

import os
import numpy as np
import librosa
import cv2
from .config import Config


class AudioProcessor:
    def __init__(self, config=None):
        self.config = config or Config()

    def load_audio(self, audio_path):
        """
        加载音频文件，支持mp3和wav格式
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 如果是mp3文件，先转换为wav
        if audio_path.lower().endswith(".mp3"):
            wav_path = os.path.splitext(audio_path)[0] + ".wav"
            if not os.path.exists(wav_path):
                print(f"转换MP3到WAV: {audio_path}")
                self._convert_mp3_to_wav(audio_path, wav_path)
            audio_path = wav_path

        # 加载音频
        print(f"加载音频: {audio_path}")
        y, sr = librosa.load(
            audio_path,
            sr=self.config.SAMPLE_RATE,
            mono=self.config.MONO,
            duration=self.config.DURATION,
        )

        return y, sr

    def _convert_mp3_to_wav(self, mp3_path, wav_path):
        """转换MP3到WAV格式"""
        import soundfile as sf

        y, sr = librosa.load(mp3_path, sr=self.config.SAMPLE_RATE)
        sf.write(wav_path, y, sr)

    def compute_mel_spectrogram(self, y, sr):
        """
        计算Mel频谱图
        """
        print("计算Mel频谱图...")

        # 计算Mel频谱
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX,
        )

        # 转换为分贝尺度
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 归一化到0-255
        mel_normalized = self._normalize_to_uint8(log_mel)

        return mel_spec, log_mel, mel_normalized

    def adaptive_binarization(self, mel_normalized):
        """
        自适应二值化Mel频谱
        """
        print("自适应二值化...")

        # 应用自适应阈值
        binary = cv2.adaptiveThreshold(
            mel_normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            self.config.ADAPTIVE_THRESHOLD_C,
        )

        # 形态学操作去除噪声
        kernel = np.ones(
            (self.config.MORPH_KERNEL_SIZE, self.config.MORPH_KERNEL_SIZE), np.uint8
        )
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 转换为0/1矩阵
        binary_matrix = (binary_cleaned > 0).astype(np.float32)

        return binary_matrix

    def calculate_energy(self, y, hop_length):
        """计算音频的RMS能量曲线并归一化"""
        rmse = librosa.feature.rms(
            y=y, frame_length=self.config.N_FFT, hop_length=hop_length
        )[0]

        # 平滑处理
        window_size = self.config.ENERGY_SMOOTHING_WINDOW
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            rmse = np.convolve(rmse, kernel, mode="same")

        # 归一化到 0-1
        rmse_min = rmse.min()
        rmse_max = rmse.max()
        if rmse_max > rmse_min:
            energy_profile = (rmse - rmse_min) / (rmse_max - rmse_min)
        else:
            energy_profile = np.zeros_like(rmse)

        return energy_profile

    def extract_note_events(self, binary_matrix, log_mel, sr, hop_length):
        """
        提取音符，新增功能：记录每个音符的能量强度(magnitude)
        注意：参数增加了 log_mel
        """
        print("提取音符事件(含强度检测)...")
        note_events = []
        n_freq_bins, n_time_frames = binary_matrix.shape
        time_per_frame = hop_length / sr

        for freq_bin in range(n_freq_bins):
            time_series = binary_matrix[freq_bin, :]
            # 找到连续激活的区域
            changes = np.diff(np.concatenate(([0], time_series, [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            for start_frame, end_frame in zip(starts, ends):
                duration_frames = end_frame - start_frame
                duration_ms = duration_frames * time_per_frame * 1000

                if (
                    duration_ms >= self.config.MIN_NOTE_DURATION_MS
                    and duration_ms <= self.config.MAX_NOTE_DURATION_MS
                ):

                    start_time = start_frame * time_per_frame * 1000
                    end_time = end_frame * time_per_frame * 1000

                    # === 新增：获取该音符在 Log-Mel 谱上的平均强度 ===
                    # 这决定了它是一个主要音符还是背景噪音
                    magnitude = np.mean(log_mel[freq_bin, start_frame:end_frame])

                    note_events.append(
                        {
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": duration_ms,
                            "frequency_bin": freq_bin,
                            "start_frame": start_frame,  # 保留帧索引用于查询能量
                            "end_frame": end_frame,
                            "magnitude": magnitude,  # <--- 关键参数
                        }
                    )

        # 按开始时间排序
        note_events.sort(key=lambda x: x["start_time"])

        print(f"提取到 {len(note_events)} 个音符事件")
        return note_events

    def _normalize_to_uint8(self, data):
        """归一化数据到0-255范围"""
        data_min = data.min()
        data_max = data.max()

        if data_max > data_min:
            normalized = 255 * (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)

        return normalized.astype(np.uint8)

    def process_audio(self, audio_path):
        """
        完整的音频处理流程
        返回: (y, sr, mel_spec, log_mel, binary_matrix, note_events)
        """
        # 1. 加载音频
        y, sr = self.load_audio(audio_path)

        # 2. 计算Mel频谱
        mel_spec, log_mel, mel_normalized = self.compute_mel_spectrogram(y, sr)

        # 3. 自适应二值化
        binary_matrix = self.adaptive_binarization(mel_normalized)

        # 4. 计算能量曲线
        energy_profile = self.calculate_energy(y, self.config.HOP_LENGTH)

        # 5. 提取音符事件 (传入 log_mel 以获取强度)
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
            "energy_profile": energy_profile,  # <--- 新增返回
            "hop_length": self.config.HOP_LENGTH,
        }
