import numpy as np


class OsuFile:
    def __init__(self, osu_path):
        self.osu_path = osu_path
        self.hit_objects = []
        self.num_columns = None
        self.original_lines = []

    def parse(self):
        """
        解析 .osu 文件，返回按键信息和轨道数。
        """
        with open(self.osu_path, "r", encoding="utf-8") as f:
            self.original_lines = f.readlines()

        in_hitobjects = False
        for line in self.original_lines:
            line = line.strip()

            if line.startswith("[Difficulty]"):
                for difficulty_line in self.original_lines:
                    difficulty_line = difficulty_line.strip()
                    if difficulty_line.startswith("CircleSize:"):
                        self.num_columns = int(
                            float(difficulty_line.split(":")[1].strip())
                        )
                        break

            if line.startswith("[HitObjects]"):
                in_hitobjects = True
                continue

            if not in_hitobjects or not line:
                continue

            # 直接保留整个行内容
            self.hit_objects.append(line)

        if self.num_columns is None:
            raise ValueError(
                "CircleSize (轨道数) 未从 [Difficulty] 部分正确读取，请检查文件格式！"
            )

    def save(self, modified_hit_objects, suffix="_modified"):
        """
        保存修改后的谱面。
        """
        new_file_path = self.osu_path.replace(".osu", f"{suffix}.osu")
        with open(new_file_path, "w", encoding="utf-8") as f:
            in_hitobjects = False
            for line in self.original_lines:
                clean_line = line.rstrip("\r\n")
                if clean_line.strip() == "[HitObjects]":
                    in_hitobjects = True
                    f.write(clean_line + "\n")
                    break
                f.write(clean_line + "\n")

            # 写入修改后的 hit_objects（逐行保持原始格式）
            for obj in modified_hit_objects:
                f.write(obj + "\n")

        print(f"Modified file saved to: {new_file_path}")

    def convert_to_pulse(self, lane_id, total_length_ms=None, fs=256):
        """
        根据 lane_id 筛选音符，生成该轨道的脉冲序列。
        lane_id: 要筛选的轨道编号。
        total_length_ms: 谱面总长度（可选）。
        fs: 采样率。
        """
        times = []
        for obj in self.hit_objects:
            parts = obj.split(",")
            if len(parts) >= 6:  # 确保有足够的字段
                try:
                    x = int(parts[0])
                    time_ms = int(parts[2])
                    lane = int(x * self.num_columns / 512)  # 计算轨道
                    if lane == lane_id:
                        times.append(time_ms)
                except ValueError:
                    continue

        if not times:
            return np.zeros(0)

        if total_length_ms is None:
            total_length_ms = max(times) + 2000  # 默认为最后一个音符时间 + 2秒

        N = int(total_length_ms * fs / 1000.0)  # 总采样点数
        signal = np.zeros(N, dtype=np.float32)

        for t in times:
            idx = int(round(t * fs / 1000.0))
            if 0 <= idx < N:
                signal[idx] = 1.0

        return signal
