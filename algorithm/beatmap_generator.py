# -*- coding: utf-8 -*-
"""
Beatmap generator for osu!mania .osu files.
"""

import os

from .config import Config
from .utils import copy_audio_to_output_dir, resolve_output_dir


class BeatmapGenerator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        self.metadata = {}
        self.difficulty = {}
        self.timing_points = []
        self.hit_objects = []

    def set_metadata(self, title, artist, creator, version, source="", tags=""):
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
        self.difficulty = {
            "HPDrainRate": hp,
            "CircleSize": cs,
            "OverallDifficulty": od,
            "ApproachRate": ar,
            "SliderMultiplier": slider_multiplier,
            "SliderTickRate": slider_tick_rate,
        }

    def generate_from_features(self, features, audio_filename):
        print("生成谱面...")

        self.metadata["AudioFilename"] = os.path.basename(audio_filename)
        bpm = features["bpm_info"]["bpm"]
        beat_length_ms = 60000 / bpm
        first_beat_time = float(
            features.get("bpm_info", {}).get("first_beat_time", 0.0)
        )
        timing_offset_ms = first_beat_time * 1000.0
        while timing_offset_ms >= beat_length_ms and beat_length_ms > 0:
            timing_offset_ms -= beat_length_ms
        while timing_offset_ms < 0.0 and beat_length_ms > 0:
            timing_offset_ms += beat_length_ms
        self.timing_points.append(
            [round(timing_offset_ms, 3), beat_length_ms, 4, 2, 1, 60, 1, 0]
        )

        for note in features["controlled_notes"]:
            self.hit_objects.append(self._create_hit_object(note))

        print(f"生成了 {len(self.hit_objects)} 个HitObjects")

    def _create_hit_object(self, note):
        x = note.get("x_position", 256)
        y = 192
        time = int(note["aligned_time"])
        duration = int(note.get("duration", 0))

        if duration >= self.config.HOLD_NOTE_MIN_DURATION:
            end_time = int(note["end_time"])
            return f"{x},{y},{time},128,0,{end_time}:0:0:0:0:"

        return f"{x},{y},{time},1,0,0:0:0:0:"

    def save(self, output_path):
        print(f"保存谱面到: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as handle:
            self._write_general_section(handle)
            self._write_editor_section(handle)
            self._write_metadata_section(handle)
            self._write_difficulty_section(handle)
            self._write_events_section(handle)
            self._write_timing_points_section(handle)
            self._write_hit_objects_section(handle)

        print(f"谱面保存完成: {output_path}")

    def _write_general_section(self, handle):
        handle.write("osu file format v14\n\n")
        handle.write("[General]\n")
        handle.write(
            f"AudioFilename: {self.metadata.get('AudioFilename', 'audio.mp3')}\n"
        )
        handle.write("AudioLeadIn: 0\n")
        handle.write("PreviewTime: -1\n")
        handle.write("Countdown: 0\n")
        handle.write("SampleSet: Soft\n")
        handle.write("StackLeniency: 0.7\n")
        handle.write("Mode: 3\n")
        handle.write("LetterboxInBreaks: 0\n")
        handle.write("SpecialStyle: 0\n")
        handle.write("WidescreenStoryboard: 0\n\n")

    def _write_editor_section(self, handle):
        handle.write("[Editor]\n")
        handle.write("Bookmarks: \n")
        handle.write("DistanceSpacing: 1.2\n")
        handle.write("BeatDivisor: 4\n")
        handle.write("GridSize: 4\n")
        handle.write("TimelineZoom: 2.3\n\n")

    def _write_metadata_section(self, handle):
        handle.write("[Metadata]\n")
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
                handle.write(f"{key}:{self.metadata[key]}\n")
        handle.write(f"BeatmapID:{self.metadata.get('BeatmapID', 0)}\n")
        handle.write(f"BeatmapSetID:{self.metadata.get('BeatmapSetID', -1)}\n\n")

    def _write_difficulty_section(self, handle):
        handle.write("[Difficulty]\n")
        for key, value in self.difficulty.items():
            handle.write(f"{key}:{value}\n")
        handle.write("\n")

    def _write_events_section(self, handle):
        handle.write("[Events]\n")
        handle.write("//Background and Video events\n")
        handle.write('0,0,"",0,0\n')
        handle.write("//Break Periods\n")
        handle.write("//Storyboard Layer 0 (Background)\n")
        handle.write("//Storyboard Layer 1 (Fail)\n")
        handle.write("//Storyboard Layer 2 (Pass)\n")
        handle.write("//Storyboard Layer 3 (Foreground)\n")
        handle.write("//Storyboard Layer 4 (Overlay)\n")
        handle.write("//Storyboard Sound Samples\n\n")

    def _write_timing_points_section(self, handle):
        handle.write("[TimingPoints]\n")
        for point in self.timing_points:
            handle.write(",".join(str(value) for value in point) + "\n")
        handle.write("\n")

    def _write_hit_objects_section(self, handle):
        handle.write("[HitObjects]\n")
        for hit_object in self.hit_objects:
            handle.write(hit_object + "\n")

    def generate_beatmap(
        self,
        audio_path,
        features,
        output_dir=None,
        iteration=None,
        output_filename=None,
        copy_audio=None,
        version_label=None,
    ):
        self.reset()
        copy_audio = (
            self.config.COPY_AUDIO_TO_OUTPUT_DIR if copy_audio is None else copy_audio
        )
        output_dir = resolve_output_dir(
            audio_path,
            output_dir=output_dir or self.config.OUTPUT_DIR,
            export_subdir=self.config.EXPORT_SUBDIR,
        )

        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        beatmap_audio_path = audio_path
        if copy_audio:
            beatmap_audio_path = copy_audio_to_output_dir(audio_path, output_dir)

        if version_label:
            version = version_label
        elif iteration is not None:
            version = f"Auto v2.1 ({self.config.DEFAULT_COLUMNS}K) - Iter{iteration}"
        else:
            version = f"Auto v2.1 ({self.config.DEFAULT_COLUMNS}K)"

        self.set_metadata(
            title=audio_basename,
            artist=audio_basename,
            creator="AutoMakeosuFile",
            version=version,
            tags="auto-generated",
        )
        self.set_difficulty(hp=5, cs=self.config.DEFAULT_COLUMNS, od=5, ar=5)
        self.generate_from_features(features, beatmap_audio_path)

        if output_filename is None:
            if iteration is not None:
                output_filename = f"{audio_basename}_iter{iteration}_{self.config.DEFAULT_COLUMNS}K.osu"
            else:
                output_filename = f"{audio_basename}_{self.config.DEFAULT_COLUMNS}K.osu"

        output_path = os.path.join(output_dir, output_filename)
        self.save(output_path)

        return output_path
