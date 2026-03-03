"""
谱面生成模块 - 生成.osu文件
"""

import os
import datetime
from .config import Config


class BeatmapGenerator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.metadata = {}
        self.difficulty = {}
        self.timing_points = []
        self.hit_objects = []

    def set_metadata(self, title, artist, creator, version, source="", tags=""):
        """设置谱面元数据"""
        self.metadata = {
            "Title": title,
            "TitleUnicode": title,
            "Artist": artist,
            "ArtistUnicode": artist,
            "Creator": creator,
            "Version": version,
            "Source": source,
            "Tags": tags,
            "BeatmapID": 0,
            "BeatmapSetID": -1,
        }

    def set_difficulty(
        self, hp=5, cs=4, od=5, ar=5, slider_multiplier=1.4, slider_tick_rate=1
    ):
        """设置难度参数"""
        self.difficulty = {
            "HPDrainRate": hp,
            "CircleSize": cs,
            "OverallDifficulty": od,
            "ApproachRate": ar,
            "SliderMultiplier": slider_multiplier,
            "SliderTickRate": slider_tick_rate,
        }

    def generate_from_features(self, features, audio_filename):
        """
        从特征数据生成谱面
        """
        print("生成谱面...")

        # 设置音频文件名
        self.metadata["AudioFilename"] = os.path.basename(audio_filename)

        # 添加TimingPoint
        bpm = features["bpm_info"]["bpm"]
        self.timing_points.append(
            [
                0,  # time
                60000 / bpm,  # beatLength
                4,  # meter
                2,  # sampleSet
                1,  # sampleIndex
                60,  # volume
                1,  # uninherited
                0,  # effects
            ]
        )

        # 生成HitObjects
        controlled_notes = features["controlled_notes"]
        num_columns = features["config"]["columns"]

        for note in controlled_notes:
            hit_object = self._create_hit_object(note, num_columns)
            self.hit_objects.append(hit_object)

        print(f"生成了 {len(self.hit_objects)} 个HitObjects")

    def _create_hit_object(self, note, num_columns):
        """
        创建单个HitObject
        """
        x = note.get("x_position", 256)  # 默认中间
        y = 192  # osu!mania固定y坐标
        time = int(note["aligned_time"])

        # 判断是单点还是长条
        duration = note.get("duration", 0)

        if duration > 100:  # 持续时间大于100ms视为长条
            end_time = int(note["end_time"])
            hit_object = f"{x},{y},{time},128,0,{end_time}:0:0:0:0:"
        else:
            hit_object = f"{x},{y},{time},1,0,0:0:0:0:"

        return hit_object

    def save(self, output_path):
        """
        保存谱面到.osu文件
        """
        print(f"保存谱面到: {output_path}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_general_section(f)
            self._write_editor_section(f)
            self._write_metadata_section(f)
            self._write_difficulty_section(f)
            self._write_events_section(f)
            self._write_timing_points_section(f)
            self._write_hit_objects_section(f)

        print(f"谱面保存完成: {output_path}")

    def _write_general_section(self, f):
        """写入General部分"""
        f.write("osu file format v14\n\n")
        f.write("[General]\n")
        f.write(f"AudioFilename: {self.metadata.get('AudioFilename', 'audio.mp3')}\n")
        f.write("AudioLeadIn: 0\n")
        f.write("PreviewTime: -1\n")
        f.write("Countdown: 0\n")
        f.write("SampleSet: Soft\n")
        f.write("StackLeniency: 0.7\n")
        f.write("Mode: 3\n")  # 3表示osu!mania
        f.write("LetterboxInBreaks: 0\n")
        f.write("SpecialStyle: 0\n")
        f.write("WidescreenStoryboard: 0\n\n")

    def _write_editor_section(self, f):
        """写入Editor部分"""
        f.write("[Editor]\n")
        f.write("Bookmarks: \n")
        f.write("DistanceSpacing: 1.2\n")
        f.write("BeatDivisor: 4\n")
        f.write("GridSize: 4\n")
        f.write("TimelineZoom: 2.3\n\n")

    def _write_metadata_section(self, f):
        """写入Metadata部分"""
        f.write("[Metadata]\n")
        for key in [
            "Title",
            "TitleUnicode",
            "Artist",
            "ArtistUnicode",
            "Creator",
            "Version",
            "Source",
            "Tags",
        ]:
            if key in self.metadata:
                f.write(f"{key}:{self.metadata[key]}\n")
        f.write(f"BeatmapID:{self.metadata.get('BeatmapID', 0)}\n")
        f.write(f"BeatmapSetID:{self.metadata.get('BeatmapSetID', -1)}\n\n")

    def _write_difficulty_section(self, f):
        """写入Difficulty部分"""
        f.write("[Difficulty]\n")
        for key, value in self.difficulty.items():
            f.write(f"{key}:{value}\n")
        f.write("\n")

    def _write_events_section(self, f):
        """写入Events部分"""
        f.write("[Events]\n")
        f.write("//Background and Video events\n")
        f.write('0,0,"",0,0\n')
        f.write("//Break Periods\n")
        f.write("//Storyboard Layer 0 (Background)\n")
        f.write("//Storyboard Layer 1 (Fail)\n")
        f.write("//Storyboard Layer 2 (Pass)\n")
        f.write("//Storyboard Layer 3 (Foreground)\n")
        f.write("//Storyboard Layer 4 (Overlay)\n")
        f.write("//Storyboard Sound Samples\n\n")

    def _write_timing_points_section(self, f):
        """写入TimingPoints部分"""
        f.write("[TimingPoints]\n")
        for point in self.timing_points:
            f.write(",".join(str(p) for p in point) + "\n")
        f.write("\n")

    def _write_hit_objects_section(self, f):
        """写入HitObjects部分"""
        f.write("[HitObjects]\n")
        for obj in self.hit_objects:
            f.write(obj + "\n")

    def generate_beatmap(self, audio_path, features, output_dir=None):
        """
        完整的谱面生成流程
        """
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR

        # 设置元数据
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        self.set_metadata(
            title=audio_basename,
            artist=audio_basename,
            creator="AutoMakeosuFile",
            version=f"Auto v2.0 ({self.config.DEFAULT_COLUMNS}K)",
            tags="auto-generated",
        )

        # 设置难度
        self.set_difficulty(
            hp=5, cs=self.config.DEFAULT_COLUMNS, od=5, ar=5  # CircleSize等于键数
        )

        # 生成谱面
        self.generate_from_features(features, audio_path)

        # 保存文件
        output_filename = f"{audio_basename}_{self.config.DEFAULT_COLUMNS}K.osu"
        output_path = os.path.join(output_dir, output_filename)
        self.save(output_path)

        # 复制到osu!歌曲目录
        self._copy_to_osu_songs_dir(audio_path, output_path, audio_basename)

        return output_path

    def _copy_to_osu_songs_dir(self, audio_path, osu_path, audio_basename):
        """
        将生成的谱面复制到osu!歌曲目录
        """
        osu_songs_dir = r"D:\osu!\Songs"

        if not os.path.exists(osu_songs_dir):
            print(f"警告: osu!歌曲目录不存在: {osu_songs_dir}")
            return

        # 创建目标文件夹
        target_folder_name = f"{audio_basename}_automake"
        target_folder = os.path.join(osu_songs_dir, target_folder_name)
        os.makedirs(target_folder, exist_ok=True)

        # 复制音频文件
        audio_filename = os.path.basename(audio_path)
        target_audio_path = os.path.join(target_folder, audio_filename)

        try:
            import shutil

            shutil.copy2(audio_path, target_audio_path)
            print(f"✓ 音频文件复制到: {target_audio_path}")
        except Exception as e:
            print(f"✗ 音频文件复制失败: {e}")

        # 复制.osu文件
        target_osu_path = os.path.join(target_folder, os.path.basename(osu_path))

        try:
            import shutil

            shutil.copy2(osu_path, target_osu_path)
            print(f"✓ 谱面文件复制到: {target_osu_path}")
        except Exception as e:
            print(f"✗ 谱面文件复制失败: {e}")

        print(f"✓ 谱面已复制到osu!歌曲目录: {target_folder}")
