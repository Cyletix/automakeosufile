import numpy as np


class DensityFixer:
    def __init__(self, bpm, fs=256, num_columns=None):
        self.bpm = bpm
        self.fs = fs
        self.num_columns = num_columns  # 轨道数

    def find_exceeding_frequencies(self, amplitude_matrices, target_frequencies):
        """
        找到超出八分音频率限制的按键。
        """
        eight_division_freq = self.bpm / 60 * 8
        exceeding_hits = []
        for track_idx, amplitude_matrix in enumerate(amplitude_matrices):
            for freq_idx, freq in enumerate(target_frequencies):
                if freq > eight_division_freq:
                    time_indices = np.where(amplitude_matrix[freq_idx] > 0)[0]
                    for time_idx in time_indices:
                        exceeding_hits.append((time_idx, track_idx))
        return exceeding_hits

    def modify_hit_objects(self, hit_objects):
        """
        修改第 0 和第 7 轨道中按键间隔小于 4 分音的按键。
        仅适用于 8K 模式。
        """
        if self.num_columns != 8:
            print(f"轨道数为 {self.num_columns}，无需处理。")
            return hit_objects

        interval_nth = 60000 / (self.bpm * 4)  # 计算 4 分音间隔（毫秒）
        print(f"4 分音间隔: {interval_nth:.2f} ms")

        modified_hit_objects = []
        last_time = {0: -float("inf"), 7: -float("inf")}  # 上次按键时间

        for obj in hit_objects:
            parts = obj.split(",")
            if len(parts) >= 6:
                try:
                    x = int(parts[0])
                    time_ms = int(parts[2])
                    lane = int(x * self.num_columns / 512)

                    # 只处理轨道 0 和 7
                    if lane in last_time:
                        if time_ms - last_time[lane] < interval_nth:
                            print(
                                f"删除按键: 轨道 {lane}, 时间 {time_ms} ms, 间隔 {time_ms - last_time[lane]:.2f} ms"
                            )
                            continue
                        last_time[lane] = time_ms
                except ValueError:
                    print(f"无法解析按键: {obj}")
                    pass

            modified_hit_objects.append(obj)

        print(
            f"修改完成，原始按键数: {len(hit_objects)}，修改后按键数: {len(modified_hit_objects)}"
        )
        return modified_hit_objects


# def find_exceeding_frequencies(amplitude_matrices, target_frequencies, bpm, fs):
#     """
#     找到超出八分音频率限制的时间点和轨道编号。
#     """
#     eight_division_freq = bpm / 60 * 8  # 八分音频率
#     exceeding_hits = []  # 保存超出频率的 (time_idx, track_idx)

#     for track_idx, amplitude_matrix in enumerate(amplitude_matrices):
#         for freq_idx, freq in enumerate(target_frequencies):
#             if freq > eight_division_freq:  # 超出八分音频率
#                 time_indices = np.where(amplitude_matrix[freq_idx] > 0)[0]
#                 for time_idx in time_indices:
#                     exceeding_hits.append((time_idx, track_idx))

#     return exceeding_hits


# def modify_hit_objects(hit_objects, exceeding_hits, fs):
#     """
#     修改谱面按键，去除超出频率的部分。
#     偶数编号按键删除。
#     """
#     # 将超出频率的时间点转换为时间范围
#     time_set = set()
#     for time_idx, track_idx in exceeding_hits:
#         # 将 time_idx 转换为时间点
#         time_set.add((time_idx * 1000 // fs, track_idx))  # 转换为毫秒

#     # 遍历 hit_objects，删除偶数编号的按键
#     modified_hit_objects = []
#     for idx, hit in enumerate(hit_objects):
#         time, lane, obj_type = hit
#         if (time, lane) in time_set:
#             # 如果在超出频率范围，偶数编号删除
#             if idx % 2 == 1:
#                 modified_hit_objects.append(hit)
#         else:
#             modified_hit_objects.append(hit)

#     return modified_hit_objects


# def save_modified_osu_file(original_lines, modified_hit_objects, osu_path):
#     """
#     将修改后的按键信息保存为新的 .osu 文件。
#     """
#     new_file_path = osu_path.replace(".osu", "_modified.osu")
#     with open(new_file_path, "w", encoding="utf-8") as f:
#         # 复制原文件的内容直到 [HitObjects]
#         in_hitobjects = False
#         for line in original_lines:
#             if line.strip() == "[HitObjects]":
#                 in_hitobjects = True
#                 f.write(line + "\n")
#                 break
#             f.write(line + "\n")

#         # 写入修改后的 hit_objects
#         for obj in modified_hit_objects:
#             time, lane, obj_type = obj
#             x = int(
#                 (lane + 0.5) * (512 / len(set([o[1] for o in modified_hit_objects])))
#             )  # 反推出 x 坐标
#             y = 192  # 固定 y 坐标
#             f.write(f"{x},{y},{time},{obj_type},0:0:0:0:\n")

#     print(f"Modified file saved to: {new_file_path}")


# def fix_and_save_density_issues(
#     osu_path, bpm, num_columns, original_lines, hit_objects
# ):
#     """
#     高级优化按键密度的接口。
#     """
#     # 调用密度检测与修复函数
#     fixed_hit_objects = advance_detect_and_fix_density(
#         hit_objects, bpm, num_keys=num_columns
#     )

#     # 保存修改后的谱面
#     save_modified_osu_file(original_lines, fixed_hit_objects, osu_path)


# def advance_detect_and_fix_density(hit_list, bpm, num_keys=4):
#     """
#     检测并修正谱面中密度过高的情况。
#     hit_list: [(time_ms, lane), ...] 已按 time_ms 排序好的按键列表。
#     bpm: 当前曲目的 bpm.
#     num_keys: 键位数量（4K/6K/7K/8K等）。
#     返回修正后的按键列表(顺序不变，但可能部分 lane 被修改)。

#     思路：对于每个按键，如果其后短时间(16分音)内，同一轨道密度过高，
#          则在剩余轨道中找到可行的一个轨道，用于移动。
#          若全部轨道都无法避免密度大，则暂时不动(这里可进一步迭代、回溯等)。
#     """
#     # 16 分音的毫秒间隔
#     # 1 分音(四分音)对应 60000 / bpm 毫秒, 16分音则要再除以4
#     interval_16th = 60000.0 / (bpm * 4)

#     # 参数: 每个 16分音内，若同轨道按键超过多少算密度过大？
#     # 这里为示例，假设阈值 threshold=2，若 16分音内同轨道出现 2 次以上就算“密度过大”
#     threshold = 2

#     # 返回值
#     fixed_list = hit_list[:]

#     for i in range(len(fixed_list)):
#         time_i, lane_i = fixed_list[i]

#         # 找到 [time_i, time_i + interval_16th) 区间内同一轨道出现的按键个数(包括自己)
#         count_same_lane = 1  # 自身算 1
#         for j in range(i + 1, len(fixed_list)):
#             time_j, lane_j = fixed_list[j]
#             if time_j - time_i > interval_16th:
#                 break
#             if lane_j == lane_i:
#                 count_same_lane += 1

#         # 如果超过阈值，尝试移动
#         if count_same_lane > threshold:
#             # 在其余轨道中尝试
#             candidate_lanes = [l for l in range(num_keys) if l != lane_i]
#             best_lane = lane_i
#             best_lane_count = count_same_lane

#             for c_lane in candidate_lanes:
#                 # 计算若把当前按键改到 c_lane，接下来 16分音范围内同轨道总数
#                 c_count = 1  # 自身也要算
#                 for j in range(i + 1, len(fixed_list)):
#                     t_j, l_j = fixed_list[j]
#                     if t_j - time_i > interval_16th:
#                         break
#                     if l_j == c_lane:
#                         c_count += 1
#                 # 比较并选择使密度最小的轨道
#                 if c_count < best_lane_count:
#                     best_lane = c_lane
#                     best_lane_count = c_count

#             # 若找到更优轨道，则移动
#             if best_lane != lane_i:
#                 fixed_list[i] = (time_i, best_lane)

#     return fixed_list


# if __name__ == "__main__":
#     # 假设我们已有 [(time_ms, lane), ...] 列表
#     # 这里示例用一小段数据
#     test_hit_list = [
#         (1000, 0),
#         (1050, 0),
#         (1100, 0),
#         (2000, 1),
#         (2050, 1),
#         (2500, 1),
#         (3000, 2),
#         (3050, 2),
#         (3100, 2),
#         (3200, 0),
#     ]
#     bpm = 150  # 示例 BPM
#     fixed_hits = advance_detect_and_fix_density(test_hit_list, bpm, num_keys=4)
#     print("原始:", test_hit_list)
#     print("修正:", fixed_hits)
