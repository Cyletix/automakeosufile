"""
Description: OSU beatmap generator
Author: Cyletix
Date: 2023-03-21 03:07:31
LastEditTime: 2023-08-04 23:16:12
FilePath: \AutoMakeosuFile\osu_file_make.py
"""

import os


class OSUGenerator:
    def __init__(self):
        self.metadata = {
            "Title": "",
            "Artist": "",
            "Creator": "",
            "Version": "",
            "AudioFilename": "",
            "PreviewTime": -1,
            "BeatmapID": 0,
            "BeatmapSetID": -1,
        }
        self.difficulty = {
            "HPDrainRate": 5,
            "CircleSize": 4,
            "OverallDifficulty": 5,
            "ApproachRate": 5,
            "SliderMultiplier": 1.4,
            "SliderTickRate": 1,
        }
        self.timing_points = []
        self.hit_objects = []

    def set_audio_file(self, filename):
        self.metadata["AudioFilename"] = os.path.basename(filename)

    def set_metadata(self, title, artist, creator, version):
        self.metadata["Title"] = title
        self.metadata["Artist"] = artist
        self.metadata["Creator"] = creator
        self.metadata["Version"] = version

    def set_difficulty(self, hp, cs, od, ar):
        self.difficulty["HPDrainRate"] = hp
        self.difficulty["CircleSize"] = cs
        self.difficulty["OverallDifficulty"] = od
        self.difficulty["ApproachRate"] = ar

    def generate_from_analysis(self, bpm, hit_times, duration):
        # Add timing point
        self.timing_points.append([0, 60000 / bpm, 4, 2, 1, 60, 1, 0])

        # Add hit objects
        for time in hit_times:
            x = 256  # Center position
            y = 192  # Center position
            self.hit_objects.append([x, y, int(time), 1, 0, "0:0:0:0:"])

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            # Write metadata
            f.write("osu file format v14\n\n")
            f.write("[General]\n")
            f.write(f"AudioFilename: {self.metadata['AudioFilename']}\n")
            f.write(f"PreviewTime: {self.metadata['PreviewTime']}\n")
            f.write("Mode: 0\n\n")

            # Write metadata section
            f.write("[Metadata]\n")
            for key, value in self.metadata.items():
                if key != "AudioFilename" and key != "PreviewTime":
                    f.write(f"{key}:{value}\n")
            f.write("\n")

            # Write difficulty settings
            f.write("[Difficulty]\n")
            for key, value in self.difficulty.items():
                f.write(f"{key}:{value}\n")
            f.write("\n")

            # Write timing points
            f.write("[TimingPoints]\n")
            for point in self.timing_points:
                f.write(",".join(map(str, point)) + "\n")
            f.write("\n")

            # Write hit objects
            f.write("[HitObjects]\n")
            for obj in self.hit_objects:
                f.write(",".join(map(str, obj)) + "\n")


# Example usage
if __name__ == "__main__":
    generator = OSUGenerator()
    generator.set_audio_file("example.mp3")
    generator.set_metadata(
        "Example Title", "Example Artist", "Example Creator", "Example Version"
    )
    generator.set_difficulty(5, 4, 5, 5)
    generator.generate_from_analysis(120, [1000, 2000, 3000], 4000)
    generator.save("example.osu")
