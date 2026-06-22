from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from automakeosufile.parsers.osu_mania import parse_osu_file


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}


@dataclass(slots=True)
class SongEntry:
    song_dir: Path
    osu_path: Path
    audio_path: Path
    mode: int
    key_count: int
    title: str
    artist: str
    version: str
    creator: str


def build_song_index(songs_root: Path) -> list[SongEntry]:
    songs_root = Path(songs_root)
    if not songs_root.exists():
        raise FileNotFoundError(f"Songs root does not exist: {songs_root}")

    entries: list[SongEntry] = []
    for osu_path in songs_root.rglob("*.osu"):
        try:
            parsed = parse_osu_file(osu_path)
        except Exception:
            continue

        if parsed.mode != 3:
            continue

        audio_filename = parsed.general.get("AudioFilename", "").strip()
        if not audio_filename:
            continue

        audio_path = osu_path.parent / audio_filename
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if not audio_path.exists():
            continue

        entries.append(
            SongEntry(
                song_dir=osu_path.parent,
                osu_path=osu_path,
                audio_path=audio_path,
                mode=parsed.mode,
                key_count=parsed.key_count,
                title=parsed.metadata.get("Title", ""),
                artist=parsed.metadata.get("Artist", ""),
                version=parsed.metadata.get("Version", ""),
                creator=parsed.metadata.get("Creator", ""),
            )
        )

    entries.sort(
        key=lambda entry: (
            entry.artist,
            entry.title,
            entry.version,
            str(entry.osu_path),
        )
    )
    return entries


def choose_sample_entry(entries: list[SongEntry]) -> SongEntry:
    if not entries:
        raise ValueError("No mania entries found in song index")

    preferred = sorted(
        entries,
        key=lambda entry: (
            abs(entry.key_count - 4),
            len(entry.version),
            len(entry.title),
        ),
    )
    return preferred[0]
