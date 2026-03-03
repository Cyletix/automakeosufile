"""
特征提取模块 - BPM检测、节拍对齐、频率到轨道映射
"""

import numpy as np
import librosa
from .config import Config


class FeatureExtractor:
    def __init__(self, config=None):
        self.config = config or Config()

    def detect_bpm(self, y, sr):
        """
        检测音频的BPM
        """
        print("检测BPM...")

        # 使用librosa检测BPM
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(tempo)

        # 计算第一个节拍时间
        if len(beats) > 0:
            first_beat_time = beats[0]
        else:
            first_beat_time = 0.0

        print(f"检测到BPM: {bpm:.1f}, 第一个节拍: {first_beat_time:.2f}s")

        return {"bpm": bpm, "first_beat_time": first_beat_time, "beats": beats}

    def align_to_beat_grid(self, note_events, bpm, first_beat_time=0):
        """
        将音符事件对齐到节拍网格
        """
        print("节拍网格对齐...")

        aligned_notes = []
        beat_interval_ms = 60000 / bpm  # 四分音间隔(ms)

        for note in note_events:
            start_time = note["start_time"]

            # 计算距离最近节拍的时间差
            beat_time = first_beat_time * 1000  # 转换为ms
            beat_index = round((start_time - beat_time) / beat_interval_ms)
            aligned_time = beat_time + beat_index * beat_interval_ms

            # 如果对齐后的时间差太大，保持原时间
            time_diff = abs(start_time - aligned_time)
            if time_diff < beat_interval_ms / 8:  # 小于32分音
                note["aligned_time"] = aligned_time
                note["beat_index"] = beat_index
                note["time_diff"] = time_diff
                aligned_notes.append(note)
            else:
                # 不对齐，但标记为未对齐
                note["aligned_time"] = start_time
                note["beat_index"] = None
                note["time_diff"] = time_diff
                aligned_notes.append(note)

        print(
            f"对齐了 {len([n for n in aligned_notes if n['beat_index'] is not None])}/{len(aligned_notes)} 个音符"
        )
        return aligned_notes

    def map_frequency_to_column(self, note_events, num_columns=7):
        """
        将频率bin映射到轨道列
        """
        print(f"频率到轨道映射 ({num_columns}K)...")

        if num_columns not in self.config.COLUMN_MAPPING:
            raise ValueError(f"不支持的键数: {num_columns}K")

        column_mapping = self.config.COLUMN_MAPPING[num_columns]
        n_mels = self.config.N_MELS

        mapped_notes = []

        for note in note_events:
            freq_bin = note["frequency_bin"]

            # 将Mel频率bin映射到音高类
            pitch_class = int((freq_bin / n_mels) * 12) % 12

            # 找到最接近的映射列
            if pitch_class in column_mapping:
                column = column_mapping.index(pitch_class)
            else:
                # 找到最接近的音高类
                distances = [abs(pitch_class - pc) for pc in column_mapping]
                closest_idx = np.argmin(distances)
                column = closest_idx

            note["pitch_class"] = pitch_class
            note["column"] = column
            note["x_position"] = self._calculate_x_position(column, num_columns)

            mapped_notes.append(note)

        return mapped_notes

    def apply_density_control(self, mapped_notes, bpm):
        """
        应用密度控制，避免同一轨道过密
        """
        print("应用密度控制...")

        if not mapped_notes:
            return mapped_notes

        # 按时间排序
        sorted_notes = sorted(mapped_notes, key=lambda x: x["aligned_time"])

        beat_interval_ms = 60000 / bpm
        controlled_notes = []
        column_last_time = {col: -1000 for col in range(self.config.DEFAULT_COLUMNS)}

        for note in sorted_notes:
            column = note["column"]
            current_time = note["aligned_time"]

            # 检查同一轨道的最小间隔
            time_since_last = current_time - column_last_time.get(column, -1000)

            if time_since_last < self.config.MAX_SAME_COLUMN_INTERVAL_MS:
                # 尝试移动到其他轨道
                original_column = column
                for alt_col in range(self.config.DEFAULT_COLUMNS):
                    if alt_col == column:
                        continue

                    alt_time_since_last = current_time - column_last_time.get(
                        alt_col, -1000
                    )
                    if alt_time_since_last >= self.config.MAX_SAME_COLUMN_INTERVAL_MS:
                        column = alt_col
                        break

                if column != original_column:
                    note["column"] = column
                    note["x_position"] = self._calculate_x_position(
                        column, self.config.DEFAULT_COLUMNS
                    )
                    note["original_column"] = original_column

            # 检查每拍音符数
            beat_start = (current_time // beat_interval_ms) * beat_interval_ms
            beat_notes = [
                n
                for n in controlled_notes
                if beat_start <= n["aligned_time"] < beat_start + beat_interval_ms
            ]

            if len(beat_notes) >= self.config.MAX_NOTES_PER_BEAT:
                # 跳过这个音符（密度太高）
                continue

            # 更新轨道最后时间并添加音符
            column_last_time[column] = current_time
            controlled_notes.append(note)

        print(f"密度控制后: {len(controlled_notes)}/{len(mapped_notes)} 个音符")
        return controlled_notes

    def _calculate_x_position(self, column, num_columns):
        """
        计算x坐标位置 (0-512)
        """
        width = 512 / num_columns
        return int(width * (column + 0.5))

    def extract_features(self, audio_data, note_events):
        """
        完整的特征提取流程
        """
        print("\n=== 特征提取开始 ===")

        # 1. 检测BPM
        bpm_info = self.detect_bpm(audio_data["audio"], audio_data["sample_rate"])

        # 2. 节拍网格对齐
        aligned_notes = self.align_to_beat_grid(
            note_events, bpm_info["bpm"], bpm_info["first_beat_time"]
        )

        # 3. 频率到轨道映射
        mapped_notes = self.map_frequency_to_column(
            aligned_notes, self.config.DEFAULT_COLUMNS
        )

        # 4. 密度控制
        controlled_notes = self.apply_density_control(mapped_notes, bpm_info["bpm"])

        print("=== 特征提取完成 ===\n")

        return {
            "bpm_info": bpm_info,
            "aligned_notes": aligned_notes,
            "mapped_notes": mapped_notes,
            "controlled_notes": controlled_notes,
            "config": {
                "columns": self.config.DEFAULT_COLUMNS,
                "sample_rate": audio_data["sample_rate"],
                "hop_length": audio_data["hop_length"],
            },
        }
