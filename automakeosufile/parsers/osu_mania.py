from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TimingPoint:
    time_ms: float
    beat_length_ms: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    effects: int

    @property
    def is_timing_control(self) -> bool:
        return self.uninherited and self.beat_length_ms > 0


@dataclass(slots=True)
class ManiaHitObject:
    lane: int
    time_ms: int
    is_hold: bool
    end_time_ms: int | None
    raw_line: str


@dataclass(slots=True)
class OsuFileData:
    path: Path
    general: dict[str, str]
    metadata: dict[str, str]
    difficulty: dict[str, str]
    timing_points: list[TimingPoint]
    hit_objects: list[ManiaHitObject]

    @property
    def mode(self) -> int:
        return int(self.general.get("Mode", 0))

    @property
    def key_count(self) -> int:
        return int(float(self.difficulty.get("CircleSize", 0)))

    @property
    def control_timing_points(self) -> list[TimingPoint]:
        return [point for point in self.timing_points if point.is_timing_control]


def parse_osu_file(path: str | Path) -> OsuFileData:
    path = Path(path)
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                sections.setdefault(current_section, [])
                continue
            if current_section is not None:
                sections.setdefault(current_section, []).append(line)

    general = _parse_key_value_section(sections.get("General", []))
    metadata = _parse_key_value_section(sections.get("Metadata", []))
    difficulty = _parse_key_value_section(sections.get("Difficulty", []))
    key_count = int(float(difficulty.get("CircleSize", 0) or 0))
    timing_points = _parse_timing_points(sections.get("TimingPoints", []))
    hit_objects = _parse_hit_objects(sections.get("HitObjects", []), key_count)

    return OsuFileData(
        path=path,
        general=general,
        metadata=metadata,
        difficulty=difficulty,
        timing_points=timing_points,
        hit_objects=hit_objects,
    )


def _parse_key_value_section(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_timing_points(lines: list[str]) -> list[TimingPoint]:
    timing_points: list[TimingPoint] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 8:
            continue

        timing_points.append(
            TimingPoint(
                time_ms=float(parts[0]),
                beat_length_ms=float(parts[1]),
                meter=int(parts[2]),
                sample_set=int(parts[3]),
                sample_index=int(parts[4]),
                volume=int(parts[5]),
                uninherited=parts[6] == "1",
                effects=int(parts[7]),
            )
        )
    return timing_points


def _parse_hit_objects(lines: list[str], key_count: int) -> list[ManiaHitObject]:
    objects: list[ManiaHitObject] = []
    if key_count <= 0:
        return objects

    for line in lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue

        x = int(parts[0])
        time_ms = int(parts[2])
        object_type = int(parts[3])
        lane = min(key_count - 1, int(x * key_count / 512))
        is_hold = bool(object_type & 128)
        end_time_ms = None

        if is_hold and len(parts) >= 6:
            hold_parts = parts[5].split(":", 1)
            try:
                end_time_ms = int(hold_parts[0])
            except ValueError:
                end_time_ms = None

        objects.append(
            ManiaHitObject(
                lane=lane,
                time_ms=time_ms,
                is_hold=is_hold,
                end_time_ms=end_time_ms,
                raw_line=line,
            )
        )

    return objects
