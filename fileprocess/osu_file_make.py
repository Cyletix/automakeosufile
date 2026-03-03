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
        """Set metadata for the beatmap."""
        self.metadata.update(
            {
                "Title": title,
                "TitleUnicode": title,
                "Artist": artist,
                "ArtistUnicode": artist,
                "Creator": creator,
                "Version": version,
                "Source": "",
                "Tags": "",
                "BeatmapID": 0,
                "BeatmapSetID": -1,
            }
        )

    def set_difficulty(self, hp, cs, od, ar):
        self.difficulty["HPDrainRate"] = hp
        self.difficulty["CircleSize"] = cs
        self.difficulty["OverallDifficulty"] = od
        self.difficulty["ApproachRate"] = ar

    def generate_from_analysis(
        self,
        bpm,
        hit_times,
        duration,
        chroma_data=None,
        sr=22050,
        hop_length=512,
        density=0.8,
        pattern_variation=0.3,
        column_count=4,
    ):
        # Add timing point
        self.timing_points.append([0, 60000 / bpm, 4, 2, 1, 60, 1, 0])

        # Add hit objects
        width = 512 / column_count
        for time in hit_times:
            if chroma_data is not None:
                # Convert time to frame index
                frame_index = int(time / 1000 * sr / hop_length)
                if frame_index < chroma_data.shape[1]:
                    # Get the chroma features for the frame
                    chroma_frame = chroma_data[:, frame_index]
                    # Find the active pitch classes
                    active_pitches = [i for i, v in enumerate(chroma_frame) if v == 1]
                    if active_pitches:
                        for pitch in active_pitches:
                            column = pitch % column_count
                            x = int(width * (column + 0.5))
                            y = 192
                            self.hit_objects.append([x, y, int(time), 1, 0, "0:0:0:0:"])
                        continue

            # Fallback to random column if no pitch is detected or no chroma data
            column = int(time / 100) % column_count
            x = int(width * (column + 0.5))
            y = 192  # Fixed y position for mania
            self.hit_objects.append([x, y, int(time), 1, 0, "0:0:0:0:"])

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            # Write metadata
            f.write("osu file format v14\n\n")
            f.write("[General]\n")
            f.write(f"AudioFilename: {self.metadata['AudioFilename']}\n")
            f.write(f"PreviewTime: {self.metadata['PreviewTime']}\n")
            f.write("Mode: 3\n")
            f.write("Countdown: 1\n")
            f.write("SampleSet: Soft\n")
            f.write("StackLeniency: 0.7\n")
            f.write("LetterboxInBreaks: 0\n")
            f.write("SpecialStyle: 0\n")
            f.write("WidescreenStoryboard: 0\n\n")

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
