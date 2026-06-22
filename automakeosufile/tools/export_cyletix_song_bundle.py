from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from automakeosufile.parsers.osu_mania import parse_osu_file


@dataclass(slots=True)
class ExportedNote:
    time_ms: int
    lane: int
    end_time_ms: int | None = None

    @property
    def is_hold(self) -> bool:
        return self.end_time_ms is not None and self.end_time_ms > self.time_ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出 CyletixMusicGame 兼容的 source cache、61K source preview 与任意 K 正式谱面"
    )
    parser.add_argument(
        "--cyletix-root", type=Path, required=True, help="CyletixMusicGame 根目录"
    )
    parser.add_argument(
        "--song-dir", type=Path, required=True, help="Cyletix songs 下的目标歌曲目录"
    )
    parser.add_argument("--audio-file", type=Path, help="可选：显式指定音频文件路径")
    parser.add_argument(
        "--target-key-count", type=int, default=7, help="正式谱目标键数，默认 7"
    )
    parser.add_argument(
        "--note-acceptance",
        type=float,
        default=1.0,
        help="传给 Cyletix source preview 的 note_acceptance",
    )
    parser.add_argument(
        "--base-name", default="", help="输出文件基础名，不传则用音频文件名推导"
    )
    parser.add_argument(
        "--preview-version", default="source", help="61K 预览谱版本名，默认 source"
    )
    parser.add_argument(
        "--final-version", default="AutoMakeosuFile", help="正式谱版本名前缀"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cyletix_root = args.cyletix_root.resolve()
    song_dir = args.song_dir.resolve()
    audio_path = (args.audio_file or _find_audio_file(song_dir)).resolve()

    payload = _load_payload_from_existing_cache(song_dir, audio_path)
    module = None
    if payload is None:
        module = _load_cyletix_export_module(cyletix_root)
        payload = module.build_payload(
            str(audio_path), note_acceptance=args.note_acceptance
        )

    metadata = _pick_metadata(song_dir, audio_path)
    base_name = args.base_name.strip() or _sanitize_filename(audio_path.stem.lower())
    preview_output = song_dir / f"{base_name}_source_preview.osu"
    final_output = song_dir / f"{base_name}_{args.target_key_count}k.osu"

    if module is None:
        module = _load_cyletix_export_module(cyletix_root)

    module.write_preview_beatmap(
        payload=payload,
        output_osu=preview_output,
        title=metadata["title"],
        artist=metadata["artist"],
        creator="Cyletix",
        version_label=args.preview_version,
        background_file=metadata["background_file"],
    )

    mapped_notes = _map_note_events_to_keys(
        payload=payload,
        target_key_count=args.target_key_count,
    )
    _write_final_beatmap(
        payload=payload,
        output_osu=final_output,
        audio_path=audio_path,
        title=metadata["title"],
        artist=metadata["artist"],
        version_label=f"{args.final_version} {args.target_key_count}K",
        key_count=args.target_key_count,
        notes=mapped_notes,
        background_file=metadata["background_file"],
    )

    result = {
        "song_dir": str(song_dir),
        "audio_path": str(audio_path),
        "source_cache_json": str(
            audio_path.with_name(f"{audio_path.stem}.cylenx_source_cache.json")
        ),
        "source_cache_npz": str(
            audio_path.with_name(f"{audio_path.stem}.cylenx_source_cache.npz")
        ),
        "source_preview_osu": str(preview_output),
        "final_osu": str(final_output),
        "target_key_count": args.target_key_count,
        "source_lane_count": int(payload.get("lane_count", 0) or 0),
        "source_note_count": int(payload.get("note_count", 0) or 0),
        "final_note_count": len(mapped_notes),
    }
    summary_path = song_dir / f"{base_name}_cyletix_export_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Source preview: {preview_output}")
    print(f"Final beatmap: {final_output}")
    print(f"Summary: {summary_path}")
    print(f"Source cache json: {result['source_cache_json']}")
    print(f"Source cache npz: {result['source_cache_npz']}")
    print(
        f"Source lanes: {result['source_lane_count']}  Final keys: {args.target_key_count}"
    )
    print(
        f"Source notes: {result['source_note_count']}  Final notes: {result['final_note_count']}"
    )
    return 0


def _load_cyletix_export_module(cyletix_root: Path):
    module_path = cyletix_root / "tools" / "export_symbolic_preview.py"
    if not module_path.exists():
        raise FileNotFoundError(f"未找到 Cyletix 导出脚本: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "cyletix_export_symbolic_preview", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_audio_file(song_dir: Path) -> Path:
    for extension in ("*.mp3", "*.ogg", "*.wav"):
        matches = sorted(song_dir.glob(extension))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"在歌曲目录中未找到音频文件: {song_dir}")


def _load_payload_from_existing_cache(song_dir: Path, audio_path: Path) -> dict | None:
    cache_json_path = song_dir / f"{audio_path.stem}.cylenx_source_cache.json"
    cache_npz_path = song_dir / f"{audio_path.stem}.cylenx_source_cache.npz"
    if not cache_json_path.exists() or not cache_npz_path.exists():
        return None

    cache_meta = json.loads(cache_json_path.read_text(encoding="utf-8"))
    with np.load(cache_npz_path, allow_pickle=False) as cached:
        pitch_axis = np.asarray(cached["pitch_midi_axis"], dtype=np.float32)
        note_events = _deserialize_note_events(cached["pitch_note_events_json"])
        bpm, first_beat_time_sec = _estimate_timing_from_song_dir(song_dir)
        return {
            "audio_path": str(audio_path),
            "source_kind": "pitch_source",
            "preview_stage": "pitch_note_events",
            "midi_min": (
                int(round(float(np.min(pitch_axis)))) if pitch_axis.size else 36
            ),
            "midi_max": (
                int(round(float(np.max(pitch_axis)))) if pitch_axis.size else 96
            ),
            "lane_count": int(pitch_axis.size) if pitch_axis.size else 61,
            "pitch_midi_min": (
                int(round(float(np.min(pitch_axis)))) if pitch_axis.size else 36
            ),
            "pitch_midi_max": (
                int(round(float(np.max(pitch_axis)))) if pitch_axis.size else 96
            ),
            "duration_ms": float(cached["audio_duration_ms"].item()),
            "sample_rate": int(cached["sample_rate"].item()),
            "hop_length": int(cached["hop_length"].item()),
            "pitch_hop_length": int(cached["pitch_hop_length"].item()),
            "source_cache_hit": True,
            "source_cache_path": str(cache_npz_path),
            "note_count": len(note_events),
            "bpm": bpm,
            "first_beat_time_sec": first_beat_time_sec,
            "stage_counts": {
                "raw_pitch_note_events": len(note_events),
                "aligned_notes": len(note_events),
                "timing_filtered_notes": len(note_events),
                "bar_pattern_notes": len(note_events),
                "density_filtered_notes": len(note_events),
                "silence_filtered_notes": len(note_events),
            },
            "note_events": note_events,
            "cache_meta": cache_meta,
        }


def _pick_metadata(song_dir: Path, audio_path: Path) -> dict[str, str]:
    osu_candidates = sorted(song_dir.glob("*.osu"))
    for candidate in osu_candidates:
        parsed = parse_osu_file(candidate)
        title = parsed.metadata.get("Title") or audio_path.stem
        artist = parsed.metadata.get("Artist") or audio_path.stem
        background_file = _extract_background_file(candidate)
        return {
            "title": title,
            "artist": artist,
            "background_file": background_file,
        }
    return {
        "title": audio_path.stem,
        "artist": audio_path.stem,
        "background_file": "",
    }


def _estimate_timing_from_song_dir(song_dir: Path) -> tuple[float, float]:
    for candidate in sorted(song_dir.glob("*.osu")):
        parsed = parse_osu_file(candidate)
        control_points = parsed.control_timing_points
        if not control_points:
            continue
        point = control_points[0]
        bpm = 60000.0 / point.beat_length_ms if point.beat_length_ms > 0 else 120.0
        first_beat_time_sec = float(point.time_ms) / 1000.0
        return bpm, first_beat_time_sec
    return 120.0, 0.0


def _extract_background_file(osu_path: Path) -> str:
    section = ""
    for raw_line in osu_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "Events" and line.startswith("0,0,"):
            parts = line.split(",")
            if len(parts) >= 3:
                return parts[2].strip().strip('"')
    return ""


def _deserialize_note_events(value) -> list[dict]:
    if isinstance(value, np.ndarray):
        raw = value.item() if value.shape == () else "".join(value.astype(str).tolist())
    else:
        raw = str(value)
    if not raw:
        return []
    payload = json.loads(raw)
    if isinstance(payload, list):
        return payload
    return []


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w\-.]+", "_", name.strip(), flags=re.UNICODE)
    return sanitized.strip("_") or "cyletix_export"


def _get_event_time_ms(event: dict, primary_key: str, fallback_key: str) -> int:
    value = event.get(primary_key)
    if value is None:
        value = event.get(fallback_key)
    if value is None:
        return 0
    return int(round(float(value)))


def _map_note_events_to_keys(
    payload: dict, target_key_count: int
) -> list[ExportedNote]:
    target_key_count = max(1, int(target_key_count))
    midi_min = int(payload.get("midi_min", 36) or 36)
    midi_max = int(payload.get("midi_max", midi_min) or midi_min)
    midi_span = max(1, midi_max - midi_min)

    merged: dict[tuple[int, int], ExportedNote] = {}
    for event in payload.get("note_events", []):
        start_time_ms = _get_event_time_ms(event, "start_time_ms", "start_time")
        end_time_ms = _get_event_time_ms(event, "end_time_ms", "end_time")
        if end_time_ms < start_time_ms:
            end_time_ms = start_time_ms
        pitch_midi = float(event.get("pitch_midi", midi_min))
        lane = int(round((pitch_midi - midi_min) / midi_span * (target_key_count - 1)))
        lane = max(0, min(target_key_count - 1, lane))
        note = ExportedNote(
            time_ms=start_time_ms,
            lane=lane,
            end_time_ms=end_time_ms if end_time_ms > start_time_ms else None,
        )
        key = (note.time_ms, note.lane)
        existing = merged.get(key)
        if existing is None:
            merged[key] = note
            continue
        existing_end = existing.end_time_ms or existing.time_ms
        note_end = note.end_time_ms or note.time_ms
        if note_end > existing_end:
            merged[key] = note

    return sorted(merged.values(), key=lambda item: (item.time_ms, item.lane))


def _write_final_beatmap(
    payload: dict,
    output_osu: Path,
    audio_path: Path,
    title: str,
    artist: str,
    version_label: str,
    key_count: int,
    notes: list[ExportedNote],
    background_file: str = "",
) -> None:
    output_osu.parent.mkdir(parents=True, exist_ok=True)
    bpm = float(payload.get("bpm", 120.0) or 120.0)
    first_beat_time_sec = float(payload.get("first_beat_time_sec", 0.0) or 0.0)
    beat_length_ms = 60000.0 / bpm if bpm > 0.0 else 500.0
    timing_offset_ms = _build_timing_offset_ms(first_beat_time_sec, bpm)

    with output_osu.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("osu file format v14\n\n")
        handle.write("[General]\n")
        handle.write(f"AudioFilename:{audio_path.name}\n")
        handle.write("AudioLeadIn:0\n")
        handle.write("PreviewTime:-1\n")
        handle.write("Countdown:0\n")
        handle.write("SampleSet:Soft\n")
        handle.write("StackLeniency:0.7\n")
        handle.write("Mode:3\n")
        handle.write("LetterboxInBreaks:0\n")
        handle.write("SpecialStyle:0\n")
        handle.write("WidescreenStoryboard:0\n\n")

        handle.write("[Editor]\n")
        handle.write("Bookmarks:\n")
        handle.write("DistanceSpacing:1.2\n")
        handle.write("BeatDivisor:4\n")
        handle.write("GridSize:4\n")
        handle.write("TimelineZoom:2.3\n\n")

        handle.write("[Metadata]\n")
        handle.write(f"Title:{title}\n")
        handle.write(f"TitleUnicode:{title}\n")
        handle.write(f"Artist:{artist}\n")
        handle.write(f"ArtistUnicode:{artist}\n")
        handle.write("Creator:AutoMakeosuFile\n")
        handle.write(f"Version:{version_label}\n")
        handle.write("Source:CYLETIX_AUTOGENERATED\n")
        handle.write("Tags:auto-generated cyletix_generated automakeosufile\n")
        handle.write("BeatmapID:0\n")
        handle.write("BeatmapSetID:-1\n\n")

        handle.write("[Difficulty]\n")
        handle.write("HPDrainRate:5\n")
        handle.write(f"CircleSize:{key_count}\n")
        handle.write("OverallDifficulty:5\n")
        handle.write("ApproachRate:5\n")
        handle.write("SliderMultiplier:1.4\n")
        handle.write("SliderTickRate:1\n\n")

        handle.write("[Events]\n")
        handle.write("//Background and Video events\n")
        if background_file:
            handle.write(f'0,0,"{Path(background_file).name}",0,0\n')
        else:
            handle.write('0,0,"",0,0\n')
        handle.write("//Break Periods\n")
        handle.write("//Storyboard Layer 0 (Background)\n")
        handle.write("//Storyboard Layer 1 (Fail)\n")
        handle.write("//Storyboard Layer 2 (Pass)\n")
        handle.write("//Storyboard Layer 3 (Foreground)\n")
        handle.write("//Storyboard Layer 4 (Overlay)\n")
        handle.write("//Storyboard Sound Samples\n\n")

        handle.write("[TimingPoints]\n")
        handle.write(f"{timing_offset_ms},{beat_length_ms},4,2,1,60,1,0\n\n")

        handle.write("[HitObjects]\n")
        for note in notes:
            x_position = _lane_to_osu_x(note.lane, key_count)
            if note.is_hold:
                handle.write(
                    f"{x_position},192,{note.time_ms},128,0,{note.end_time_ms}:0:0:0:0:\n"
                )
            else:
                handle.write(f"{x_position},192,{note.time_ms},1,0,0:0:0:0:\n")


def _lane_to_osu_x(lane: int, columns: int) -> int:
    lane_width = 512.0 / max(columns, 1)
    return int(round(float(lane) * lane_width + lane_width * 0.5))


def _build_timing_offset_ms(first_beat_time_sec: float, bpm: float) -> float:
    if bpm <= 0.0:
        return 0.0
    beat_length_ms = 60000.0 / float(bpm)
    timing_offset_ms = float(first_beat_time_sec) * 1000.0
    while timing_offset_ms >= beat_length_ms:
        timing_offset_ms -= beat_length_ms
    while timing_offset_ms < 0.0:
        timing_offset_ms += beat_length_ms
    return round(timing_offset_ms, 3)


if __name__ == "__main__":
    raise SystemExit(main())
