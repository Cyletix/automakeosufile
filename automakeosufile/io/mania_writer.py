from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from automakeosufile.parsers.osu_mania import OsuFileData, TimingPoint


@dataclass(slots=True)
class ManiaNote:
    time_ms: int
    lane: int
    end_time_ms: int | None = None

    @property
    def is_hold(self) -> bool:
        return self.end_time_ms is not None and self.end_time_ms > self.time_ms


class ManiaBeatmapWriter:
    def write_from_reference(
        self,
        output_path: str | Path,
        reference: OsuFileData,
        notes: list[ManiaNote],
        audio_filename: str,
        key_count: int,
        version_name: str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        general = dict(reference.general)
        metadata = dict(reference.metadata)
        difficulty = dict(reference.difficulty)

        general["AudioFilename"] = audio_filename
        general["Mode"] = "3"
        metadata["Version"] = version_name
        metadata["Creator"] = metadata.get("Creator", "AutoMakeosuFile") + " + RuleBaseline"
        difficulty["CircleSize"] = str(key_count)

        hit_object_lines = [self._format_hit_object(note, key_count) for note in notes]
        timing_point_lines = [self._format_timing_point(point) for point in reference.timing_points]

        with output_path.open("w", encoding="utf-8") as file:
            file.write("osu file format v14\n\n")
            self._write_key_value_section(file, "General", general)
            self._write_key_value_section(file, "Metadata", metadata)
            self._write_key_value_section(file, "Difficulty", difficulty)
            file.write("[TimingPoints]\n")
            for line in timing_point_lines:
                file.write(f"{line}\n")
            file.write("\n")
            file.write("[HitObjects]\n")
            for line in hit_object_lines:
                file.write(f"{line}\n")

        return output_path

    def _write_key_value_section(self, file, section_name: str, values: dict[str, str]) -> None:
        file.write(f"[{section_name}]\n")
        for key, value in values.items():
            file.write(f"{key}:{value}\n")
        file.write("\n")

    def _format_timing_point(self, point: TimingPoint) -> str:
        uninherited = 1 if point.uninherited else 0
        return ",".join(
            [
                self._fmt_float(point.time_ms),
                self._fmt_float(point.beat_length_ms),
                str(point.meter),
                str(point.sample_set),
                str(point.sample_index),
                str(point.volume),
                str(uninherited),
                str(point.effects),
            ]
        )

    def _format_hit_object(self, note: ManiaNote, key_count: int) -> str:
        x = int((512 / key_count) * (note.lane + 0.5))
        y = 192
        if note.is_hold:
            return f"{x},{y},{note.time_ms},128,0,{note.end_time_ms}:0:0:0:0:"
        return f"{x},{y},{note.time_ms},1,0,0:0:0:0:"

    def _fmt_float(self, value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.15f}".rstrip("0").rstrip(".")