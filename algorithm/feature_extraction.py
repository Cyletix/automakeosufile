"""
特征提取模块 - BPM检测、多级节拍对齐、轨道映射与物理手感修正
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
        # 移除 tight=True 参数，因为librosa.beat.beat_track()不支持
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

        # 简单处理：如果BPM是两倍速或半速的问题暂时忽略，依赖librosa
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
        将音符事件对齐到多级节拍网格 (1/1, 1/2, 1/4, 1/8, 1/3, 1/6)
        解决"Metronome"问题，捕捉更多细节
        """
        print("多级节拍网格对齐...")

        aligned_notes = []
        # 四分音符时长 (ms)
        beat_duration_ms = 60000 / bpm

        # 将第一个节拍时间转换为ms
        first_beat_ms = first_beat_time * 1000

        valid_count = 0

        for note in note_events:
            start_time = note["start_time"]

            # 计算相对于第一个节拍的时间差
            relative_time = start_time - first_beat_ms

            # 将时间差转换为"拍数" (beats)
            raw_beat_pos = relative_time / beat_duration_ms

            best_snap = None
            min_error = float("inf")
            best_divisor = 1

            # 尝试所有支持的细分 (1/1, 1/2, 1/4, 1/3 等)
            for divisor in self.config.BEAT_DIVISORS:
                # 当前细分下的吸附位置
                # 例如 divisor=4 (16分音)，我们将拍数乘4，四舍五入，再除以4
                snapped_pos = round(raw_beat_pos * divisor) / divisor

                # 计算误差 (ms)
                error_ms = abs(raw_beat_pos - snapped_pos) * beat_duration_ms

                if error_ms < min_error:
                    min_error = error_ms
                    best_snap = snapped_pos
                    best_divisor = divisor

            # 只有误差在允许范围内才吸附，否则保留原位或者可以考虑丢弃
            if min_error <= self.config.MAX_ALIGN_ERROR_MS:
                aligned_time = first_beat_ms + best_snap * beat_duration_ms
                note["aligned_time"] = aligned_time
                note["snap_divisor"] = best_divisor  # 记录是几分音，后续可用于加重音效
                valid_count += 1
            else:
                # 误差太大，不吸附，保持原位 (或者可以选择丢弃噪音)
                note["aligned_time"] = start_time
                note["snap_divisor"] = 0

            # 同步更新结束时间（保持时长不变，或者也需要对齐结束时间，这里暂时保持时长）
            note["end_time"] = note["aligned_time"] + note["duration"]
            aligned_notes.append(note)

        print(f"对齐完成: {valid_count}/{len(note_events)} 个音符成功吸附到网格")
        return aligned_notes

    def map_frequency_to_column(self, note_events, num_columns=7):
        """
        将频率bin映射到轨道列
        """
        # ... (保持原有逻辑不变，省略以节省篇幅) ...
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

        
        # 轨道平衡调整
        if hasattr(self.config, 'COLUMN_BALANCE_TARGET_STD'):
            # 计算当前轨道分布
            column_counts = {}
            for note in mapped_notes:
                col = note["column"]
                column_counts[col] = column_counts.get(col, 0) + 1
            
            # 计算标准差
            if column_counts:
                mean_count = sum(column_counts.values()) / len(column_counts)
                variance = sum((count - mean_count) ** 2 for count in column_counts.values()) / len(column_counts)
                current_std = (variance ** 0.5) / mean_count * 100 if mean_count > 0 else 0
                
                # 如果标准差超过阈值，重新平衡
                if current_std > self.config.COLUMN_BALANCE_TARGET_STD:
                    print(f"轨道不平衡 (标准差: {current_std:.1f}%)，进行重新平衡...")
                    # 简单的重新平衡：将过多的音符移动到较少的轨道
                    target_count = int(mean_count)
                    for col in list(column_counts.keys()):
                        if column_counts[col] > target_count * 1.5:  # 超过150%
                            excess = column_counts[col] - target_count
                            # 找到需要音符的轨道
                            for target_col in range(self.config.DEFAULT_COLUMNS):
                                if target_col not in column_counts or column_counts.get(target_col, 0) < target_count * 0.5:
                                    # 移动一些音符
                                    moved = 0
                                    for note in mapped_notes:
                                        if note["column"] == col and moved < excess:
                                            note["column"] = target_col
                                            note["x_position"] = self._calculate_x_position(target_col, self.config.DEFAULT_COLUMNS)
                                            moved += 1
                                            if moved >= excess:
                                                break
                                    break
        return mapped_notes

    def enforce_physical_limits(self, mapped_notes):
        """
        [新增关键函数] 强制执行物理间隔限制
        解决"间隔太小"问题，修正长条和删除过近的单点
        """
        print("执行物理手感修正 (Gap Enforcement)...")

        if not mapped_notes:
            return []

        # 1. 按轨道分组
        columns_notes = {}
        for note in mapped_notes:
            col = note["column"]
            if col not in columns_notes:
                columns_notes[col] = []
            columns_notes[col].append(note)

        final_notes = []
        min_gap = self.config.MIN_COLUMN_GAP_MS

        # 2. 遍历每个轨道处理冲突
        for col, notes in columns_notes.items():
            # 按时间排序
            notes.sort(key=lambda x: x["aligned_time"])

            processed_column = []
            if not notes:
                continue

            # 放入第一个音符
            processed_column.append(notes[0])

            for i in range(1, len(notes)):
                current_note = notes[i]
                prev_note = processed_column[-1]  # 获取前一个已确认的音符

                # 计算间隔：当前音符开始时间 - 前一个音符结束时间
                gap = current_note["aligned_time"] - prev_note["end_time"]

                if gap < min_gap:
                    # 间隔不足！需要处理

                    # 情况A: 前一个是长条 (Long Note)
                    if prev_note["duration"] > self.config.HOLD_NOTE_MIN_DURATION:  # 使用配置的长条最小持续时间
                        # 缩短前一个长条的尾巴
                        new_end_time = current_note["aligned_time"] - min_gap
                        new_duration = new_end_time - prev_note["aligned_time"]

                        if new_duration > 30:  # 如果缩短后还有长度，保留修改
                            prev_note["end_time"] = new_end_time
                            prev_note["duration"] = new_duration
                            processed_column.append(current_note)  # 添加当前音符
                        else:
                            # 缩得太短了，变成了单点
                            # 如果前一个变成单点后仍然离得很近，就需要删除前一个
                            # 这里策略是：直接删除前一个长条，保留当前的音准
                            processed_column.pop()
                            processed_column.append(current_note)

                    # 情况B: 前一个是单点 (Tap)
                    else:
                        # 按照用户要求：删除前一个，保留当前的
                        processed_column.pop()
                        processed_column.append(current_note)
                else:
                    # 间隔足够，直接添加
                    processed_column.append(current_note)

            final_notes.extend(processed_column)

        # 重新按时间排序所有音符
        final_notes.sort(key=lambda x: x["aligned_time"])
        print(
            f"物理修正后: {len(final_notes)}/{len(mapped_notes)} 个音符 (删除了 {len(mapped_notes)-len(final_notes)} 个冲突)"
        )

        return final_notes

    def _calculate_x_position(self, column, num_columns):
        """
        计算x坐标位置 (0-512)
        """
        width = 512 / num_columns
        return int(width * (column + 0.5))

    def apply_dynamic_density_filter(
        self, aligned_notes, energy_profile, hop_length, sr
    ):
        """
        根据音频能量动态过滤音符
        策略：
        1. 获取当前时刻的能量值
        2. 决定允许的节拍细分 (Snap Divisor)
        3. 决定目标密度 (Notes Per Second)
        4. 如果局部密度超标，优先保留 magnitude 大的音符
        """
        print("应用动态密度控制...")

        if not aligned_notes:
            return []

        filtered_notes = []

        # 为了高效处理，我们按时间窗口（例如每500ms）处理
        window_size_ms = 500
        current_window_start = 0
        window_notes = []

        # 先按开始时间排序
        sorted_notes = sorted(aligned_notes, key=lambda x: x["aligned_time"])

        # 辅助函数：处理一个窗口内的音符
        def process_window(notes, start_ms):
            if not notes:
                return []

            # 1. 获取该窗口的平均能量
            mid_time_ms = start_ms + window_size_ms / 2
            frame_idx = int((mid_time_ms / 1000) * sr / hop_length)
            frame_idx = min(frame_idx, len(energy_profile) - 1)
            local_energy = energy_profile[frame_idx]

            # 2. 过滤规则A: 节拍细分限制 (Snap Restrictions)
            allowed_snaps = [1]  # 默认至少允许4分音
            for thresh, snaps in self.config.SNAP_RESTRICTIONS:
                if local_energy >= thresh:
                    allowed_snaps = snaps
                else:
                    break

            # 剔除不允许的细分音符 (例如低能量时剔除1/8音)
            valid_snap_notes = []
            for n in notes:
                divisor = n.get("snap_divisor", 0)
                # 0表示未吸附，1表示4分音。如果 divisor 在允许列表中，或者它是主拍(1)，保留
                if divisor in allowed_snaps or divisor == 1:
                    valid_snap_notes.append(n)
                # 也可以选择不删除，而是强制降级（Snap到最近的允许网格），这里选择删除以降低难度

            # 3. 过滤规则B: 密度上限 (Density Cap)
            max_nps = 4.0  # 默认值，会被DENSITY_MAPPING覆盖
            for thresh, nps in self.config.DENSITY_MAPPING:
                if local_energy >= thresh:
                    max_nps = nps
                else:
                    break

            max_notes_in_window = int(max_nps * (window_size_ms / 1000))
            max_notes_in_window = max(1, max_notes_in_window)  # 至少保留1个

            # 如果音符数量超过上限，按 magnitude (音量强度) 排序，保留最强的
            if len(valid_snap_notes) > max_notes_in_window:
                # 降序排列
                valid_snap_notes.sort(key=lambda x: x.get("magnitude", 0), reverse=True)
                return valid_snap_notes[:max_notes_in_window]

            return valid_snap_notes

        # 遍历所有音符进行窗口化处理
        for note in sorted_notes:
            if note["aligned_time"] > current_window_start + window_size_ms:
                # 结算上一个窗口
                filtered_notes.extend(
                    process_window(window_notes, current_window_start)
                )
                # 移动窗口
                window_notes = []
                current_window_start += window_size_ms
                while note["aligned_time"] > current_window_start + window_size_ms:
                    current_window_start += window_size_ms

            window_notes.append(note)

        # 结算最后一个窗口
        filtered_notes.extend(process_window(window_notes, current_window_start))

        # 重新排序
        filtered_notes.sort(key=lambda x: x["aligned_time"])
        print(
            f"动态密度过滤: {len(aligned_notes)} -> {len(filtered_notes)} (保留率 {len(filtered_notes)/len(aligned_notes):.1%})"
        )

        return filtered_notes

    def extract_features(self, audio_data, note_events):
        """
        完整的特征提取流程
        """
        print("\n=== 特征提取开始 ===")

        # 1. 检测BPM
        bpm_info = self.detect_bpm(audio_data["audio"], audio_data["sample_rate"])

        # 2. 多级节拍对齐 (解决 Metronome 问题)
        aligned_notes = self.align_to_beat_grid(
            note_events, bpm_info["bpm"], bpm_info["first_beat_time"]
        )

        # === 插入新步骤：动态密度控制 ===
        # 此时 aligned_notes 包含了很多可能的音符，我们在映射到轨道前先筛一遍
        density_filtered_notes = self.apply_dynamic_density_filter(
            aligned_notes,
            audio_data["energy_profile"],
            audio_data["hop_length"],
            audio_data["sample_rate"],
        )

        # 3. 频率到轨道映射 (使用过滤后的音符)
        mapped_notes = self.map_frequency_to_column(
            density_filtered_notes, self.config.DEFAULT_COLUMNS
        )

        # 4. 物理手感修正 (解决间隔过小问题)
        # 替代了原来的 apply_density_control，因为那个逻辑是"移轨道"，我们需要的是"砍长度/删音符"
        # 你也可以保留 density_control 作为最后一道防线，但 enforce_physical_limits 必须先做
        physically_correct_notes = self.enforce_physical_limits(mapped_notes)

        print("=== 特征提取完成 ===\n")

        return {
            "bpm_info": bpm_info,
            "aligned_notes": aligned_notes,  # 调试用
            "mapped_notes": mapped_notes,  # 调试用
            "controlled_notes": physically_correct_notes,  # 最终给生成器的
            "config": {
                "columns": self.config.DEFAULT_COLUMNS,
                "sample_rate": audio_data["sample_rate"],
                "hop_length": audio_data["hop_length"],
            },
        }
