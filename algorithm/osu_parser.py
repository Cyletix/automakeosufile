import os
import re
from typing import Dict, List, Tuple
import numpy as np


class OsuFileParser:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.hit_objects = []
        self.timing_points = []
        self.metadata = {}
        self.difficulty = {}

    def parse_file(self, filepath: str) -> Dict:
        """
        解析.osu文件并提取统计信息

        返回:
            包含统计信息的字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        print(f"解析文件: {filepath}")
        self._reset()

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析各个部分
        self._parse_metadata(content)
        self._parse_difficulty(content)
        self._parse_timing_points(content)
        self._parse_hit_objects(content)

        # 计算统计信息
        stats = self._calculate_statistics()

        return stats

    def _parse_metadata(self, content: str):
        """解析Metadata部分"""
        metadata_section = self._extract_section(content, "Metadata")
        if metadata_section:
            for line in metadata_section.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    self.metadata[key.strip()] = value.strip()

    def _parse_difficulty(self, content: str):
        """解析Difficulty部分"""
        difficulty_section = self._extract_section(content, "Difficulty")
        if difficulty_section:
            for line in difficulty_section.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    self.difficulty[key.strip()] = float(value.strip())

    def _parse_timing_points(self, content: str):
        """解析TimingPoints部分"""
        timing_section = self._extract_section(content, "TimingPoints")
        if timing_section:
            for line in timing_section.split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            time = int(float(parts[0]))
                            beat_length = float(parts[1])
                            self.timing_points.append(
                                {"time": time, "beat_length": beat_length}
                            )
                        except:
                            pass

    def _parse_hit_objects(self, content: str):
        """解析HitObjects部分"""
        hitobjects_section = self._extract_section(content, "HitObjects")
        if hitobjects_section:
            for line in hitobjects_section.split("\n"):
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        try:
                            x = int(parts[0])
                            y = int(parts[1])
                            time = int(parts[2])
                            type_flags = int(parts[3])

                            hit_object = {
                                "x": x,
                                "y": y,
                                "time": time,
                                "type": type_flags,
                                "is_hold": False,
                                "end_time": time,
                            }

                            # 检查是否是长条
                            if type_flags & 128:  # 长条类型
                                if len(parts) >= 6:
                                    end_time_str = parts[5].split(":")[0]
                                    hit_object["end_time"] = int(end_time_str)
                                    hit_object["is_hold"] = True

                            self.hit_objects.append(hit_object)
                        except:
                            pass

    def _extract_section(self, content: str, section_name: str) -> str:
        """提取指定部分的内容"""
        pattern = rf"\[{section_name}\](.*?)(?=\n\[|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _calculate_statistics(self) -> Dict:
        """计算统计信息"""
        if not self.hit_objects:
            return {}

        # 按时间排序
        sorted_objects = sorted(self.hit_objects, key=lambda x: x["time"])

        # 基本统计
        total_notes = len(sorted_objects)
        hold_notes = sum(1 for obj in sorted_objects if obj["is_hold"])
        hold_percentage = (hold_notes / total_notes * 100) if total_notes > 0 else 0

        # 时间范围
        start_time = sorted_objects[0]["time"] if sorted_objects else 0
        end_time = sorted_objects[-1]["time"] if sorted_objects else 0
        total_duration_ms = end_time - start_time
        total_duration_sec = total_duration_ms / 1000

        # NPS (音符每秒)
        nps = total_notes / total_duration_sec if total_duration_sec > 0 else 0

        # 长条平均持续时间
        hold_durations = []
        for obj in sorted_objects:
            if obj["is_hold"]:
                duration = obj["end_time"] - obj["time"]
                hold_durations.append(duration)
        mean_hold_duration = np.mean(hold_durations) if hold_durations else 0

        columns = int(self.difficulty.get("CircleSize", 7) or 7)

        # 轨道分布 (osu!mania中x坐标对应轨道)
        column_distribution = {}
        column_balance = {}

        for obj in sorted_objects:
            x = obj["x"]
            column = self._x_to_column(x, columns)
            column_distribution[column] = column_distribution.get(column, 0) + 1

        # 计算轨道平衡百分比
        for col in range(columns):
            count = column_distribution.get(col, 0)
            column_balance[col] = (count / total_notes * 100) if total_notes > 0 else 0

        # 轨道平衡标准差
        balance_values = list(column_balance.values())
        column_balance_std = np.std(balance_values) if balance_values else 0

        # 音符间隔统计
        intervals = []
        for i in range(1, len(sorted_objects)):
            interval = sorted_objects[i]["time"] - sorted_objects[i - 1]["time"]
            intervals.append(interval)

        interval_stats = {
            "mean": np.mean(intervals) if intervals else 0,
            "median": np.median(intervals) if intervals else 0,
            "std": np.std(intervals) if intervals else 0,
            "min": min(intervals) if intervals else 0,
            "max": max(intervals) if intervals else 0,
        }

        return {
            "total_notes": total_notes,
            "total_duration_ms": total_duration_ms,
            "total_duration_sec": total_duration_sec,
            "nps": nps,
            "start_time": start_time,
            "end_time": end_time,
            "hold_notes_count": hold_notes,
            "hold_notes_percentage": hold_percentage,
            "mean_hold_duration": mean_hold_duration,
            "column_distribution": column_distribution,
            "column_balance": column_balance,
            "column_balance_std": column_balance_std,
            "columns": columns,
            "interval_stats": interval_stats,
            "intervals": intervals,
        }

    def _x_to_column(self, x: int, columns: int = 7) -> int:
        """将x坐标转换为轨道编号"""
        # osu!mania中x坐标范围是0-512
        column_width = 512 / columns
        column = int(x / column_width)
        return min(column, columns - 1)


def parse_osu_file(filepath: str) -> Dict:
    """解析.osu文件的便捷函数"""
    parser = OsuFileParser()
    return parser.parse_file(filepath)
