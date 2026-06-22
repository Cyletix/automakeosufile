main.py
```python
"""当前推荐入口：运行单曲检查而不是旧的实验脚本。"""

from automakeosufile.tools.inspect_sample import main


if __name__ == "__main__":
    raise SystemExit(main())
```

---

automakeosufile/__init__.py
```python
"""AutoMakeosuFile 的新主流程包。"""

from .config import FeatureConfig, dense_onset_config

__all__ = ["FeatureConfig", "dense_onset_config"]
```

---

automakeosufile/alignment/__init__.py
```python
__all__: list[str] = []
```

---

automakeosufile/alignment/offset_aligner.py
```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from automakeosufile.config import FeatureConfig, dense_onset_config
from automakeosufile.features.extractor import ExtractedFeatures, extract_features
from automakeosufile.output_paths import INSPECT_OUTPUT_DIR
from automakeosufile.parsers.osu_mania import OsuFileData, parse_osu_file


@dataclass(slots=True)
class AlignmentResult:
    offset_frames: int
    offset_ms: float
    correlation_score: float
    aligned_times_ms: list[float]
    original_times_ms: list[float]


def estimate_best_offset(
    reference_times_ms: list[int] | list[float],
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_length: int,
    max_offset_ms: float = 500.0,
) -> AlignmentResult:
    if onset_envelope.size == 0:
        raise ValueError("onset_envelope 为空，无法估计 offset")
    if not reference_times_ms:
        raise ValueError("参考谱面没有音符时间，无法估计 offset")

    frame_count = int(onset_envelope.shape[0])
    reference_frames = np.clip(
        np.rint(
            np.asarray(reference_times_ms, dtype=np.float64)
            / 1000.0
            * sample_rate
            / hop_length
        ).astype(int),
        0,
        max(frame_count - 1, 0),
    )
    reference_signal = np.zeros(frame_count, dtype=np.float64)
    np.add.at(reference_signal, reference_frames, 1.0)
    reference_signal = _smooth_reference_signal(reference_signal)

    onset_signal = np.asarray(onset_envelope, dtype=np.float64)
    onset_signal = _normalize_signal(onset_signal)
    reference_signal = _normalize_signal(reference_signal)

    max_offset_frames = max(
        1, int(round(max_offset_ms / 1000.0 * sample_rate / hop_length))
    )
    best_offset_frames = 0
    best_score = float("-inf")

    for lag in range(-max_offset_frames, max_offset_frames + 1):
        score = _lagged_dot(onset_signal, reference_signal, lag)
        if score > best_score:
            best_score = score
            best_offset_frames = lag

    offset_ms = best_offset_frames * hop_length / sample_rate * 1000.0
    aligned_times_ms = [float(time_ms + offset_ms) for time_ms in reference_times_ms]
    return AlignmentResult(
        offset_frames=best_offset_frames,
        offset_ms=float(offset_ms),
        correlation_score=float(best_score),
        aligned_times_ms=aligned_times_ms,
        original_times_ms=[float(time_ms) for time_ms in reference_times_ms],
    )


def align_osu_to_audio(
    osu_path: str | Path,
    audio_path: str | Path | None = None,
    config: FeatureConfig | None = None,
    max_offset_ms: float = 500.0,
) -> tuple[AlignmentResult, OsuFileData, ExtractedFeatures]:
    config = config or FeatureConfig()
    osu_data = parse_osu_file(osu_path)
    if audio_path is None:
        audio_path = Path(osu_path).parent / osu_data.general["AudioFilename"]

    features = extract_features(audio_path, config=config)
    gt_times_ms = [obj.time_ms for obj in osu_data.hit_objects]
    result = estimate_best_offset(
        reference_times_ms=gt_times_ms,
        onset_envelope=features.onset_envelope,
        sample_rate=features.sample_rate,
        hop_length=config.hop_length,
        max_offset_ms=max_offset_ms,
    )
    return result, osu_data, features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 GT 音符时间与 onset_envelope 互相关估计最佳 offset"
    )
    parser.add_argument("--osu", type=Path, required=True, help="参考 osu!mania 谱面")
    parser.add_argument("--audio", type=Path, help="可选：显式指定音频路径")
    parser.add_argument(
        "--dense-onset",
        action="store_true",
        help="使用高密度 onset 预设来计算 onset_envelope",
    )
    parser.add_argument(
        "--max-offset-ms",
        type=float,
        default=500.0,
        help="互相关搜索的最大偏移范围（毫秒），默认 500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INSPECT_OUTPUT_DIR / "alignment",
        help="输出 offset_alignment.json 的目录，默认 output/inspect/alignment",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = dense_onset_config() if args.dense_onset else FeatureConfig()
    result, osu_data, features = align_osu_to_audio(
        osu_path=args.osu,
        audio_path=args.audio,
        config=config,
        max_offset_ms=args.max_offset_ms,
    )

    output_dir = args.output_dir / _safe_stem(args.osu)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "offset_alignment.json"
    payload = {
        "osu_path": str(args.osu),
        "audio_path": str(features.audio_path),
        "audio_filename": osu_data.general.get("AudioFilename", ""),
        "note_count": len(osu_data.hit_objects),
        "sample_rate": features.sample_rate,
        "hop_length": config.hop_length,
        "max_offset_ms": args.max_offset_ms,
        "alignment": asdict(result),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Best offset: {result.offset_ms:.2f} ms ({result.offset_frames} frames)")
    print(f"Correlation score: {result.correlation_score:.4f}")
    print(f"Aligned note count: {len(result.aligned_times_ms)}")
    print(f"Saved alignment: {output_path}")
    return 0


def _lagged_dot(
    onset_signal: np.ndarray, reference_signal: np.ndarray, lag: int
) -> float:
    if lag > 0:
        return float(np.dot(onset_signal[lag:], reference_signal[:-lag]))
    if lag < 0:
        return float(np.dot(onset_signal[:lag], reference_signal[-lag:]))
    return float(np.dot(onset_signal, reference_signal))


def _normalize_signal(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    scale = np.linalg.norm(centered)
    if scale <= 1e-12:
        return centered
    return centered / scale


def _smooth_reference_signal(reference_signal: np.ndarray) -> np.ndarray:
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float64)
    return np.convolve(reference_signal, kernel, mode="same")


def _safe_stem(path: Path) -> str:
    name = path.stem
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
```

---

automakeosufile/baseline/__init__.py
```python
"""规则基线模块包。"""

__all__: list[str] = []
```

---

automakeosufile/baseline/rule_based_mapper.py
```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from automakeosufile.config import FeatureConfig, dense_onset_config
from automakeosufile.features.extractor import ExtractedFeatures, extract_features
from automakeosufile.io.mania_writer import ManiaBeatmapWriter, ManiaNote
from automakeosufile.output_paths import BEATMAP_OUTPUT_DIR
from automakeosufile.parsers.osu_mania import OsuFileData, TimingPoint, parse_osu_file
from automakeosufile.tools.inspect_sample import _apply_onset_overrides, _safe_stem


@dataclass(slots=True)
class MappingResult:
    output_path: Path
    note_count: int
    snapped_note_count: int


class RuleBasedMapper:
    def __init__(
        self,
        config: FeatureConfig,
        key_count: int = 6,
        smooth_frames: int = 1,
        chord_relative_threshold: float = 0.7,
        chord_absolute_threshold_ratio: float = 0.35,
        chord_absolute_percentile: float = 75.0,
        max_chord_size: int = 3,
        hold_min_duration_ms: int = 180,
        hold_relative_threshold: float = 0.45,
        hold_stability_threshold: float = 0.60,
    ):
        self.config = config
        self.key_count = key_count
        self.smooth_frames = max(0, smooth_frames)
        self.chord_relative_threshold = chord_relative_threshold
        self.chord_absolute_threshold_ratio = chord_absolute_threshold_ratio
        self.chord_absolute_percentile = chord_absolute_percentile
        self.max_chord_size = max(1, max_chord_size)
        self.hold_min_duration_ms = hold_min_duration_ms
        self.hold_relative_threshold = hold_relative_threshold
        self.hold_stability_threshold = hold_stability_threshold

    def generate_notes(
        self, osu_data: OsuFileData, features: ExtractedFeatures
    ) -> list[ManiaNote]:
        control_points = sorted(
            osu_data.control_timing_points, key=lambda point: point.time_ms
        )
        if not control_points:
            raise ValueError("Reference beatmap has no uninherited timing points")

        raw_times_ms = [float(value) for value in features.onset_times * 1000]
        snapped_times_ms = [
            self._snap_time_ms(value, control_points) for value in raw_times_ms
        ]
        lane_energy = self._lane_energy_matrix(features.cqt_magnitude)
        absolute_lane_threshold = self._absolute_lane_threshold(lane_energy)

        notes: list[ManiaNote] = []
        used_pairs: set[tuple[int, int]] = set()
        used_lanes_per_time: dict[int, set[int]] = {}

        for snapped_time_ms in snapped_times_ms:
            frame_index = self._frame_index(snapped_time_ms, features)
            smoothed_lane_scores = self._smoothed_lane_scores(lane_energy, frame_index)
            lane_scores = self._lane_scores(smoothed_lane_scores)
            selected_lanes = self._select_lanes(
                snapped_time_ms=snapped_time_ms,
                lane_scores=lane_scores,
                used_lanes_per_time=used_lanes_per_time,
                absolute_lane_threshold=absolute_lane_threshold,
            )

            for lane in selected_lanes:
                pair = (snapped_time_ms, lane)
                if pair in used_pairs:
                    continue
                used_pairs.add(pair)
                used_lanes_per_time.setdefault(snapped_time_ms, set()).add(lane)
                hold_end_time_ms = self._estimate_hold_end_time_ms(
                    start_time_ms=snapped_time_ms,
                    lane=lane,
                    start_frame_index=frame_index,
                    lane_energy=lane_energy,
                    local_peak_score=smoothed_lane_scores[lane],
                    features=features,
                    control_points=control_points,
                )
                notes.append(
                    ManiaNote(
                        time_ms=snapped_time_ms,
                        lane=lane,
                        end_time_ms=hold_end_time_ms,
                    )
                )

        notes.sort(key=lambda note: (note.time_ms, note.lane))
        return notes

    def _snap_time_ms(self, time_ms: float, control_points: list[TimingPoint]) -> int:
        timing_point = self._active_timing_point(control_points, time_ms)
        grid_spacing_ms = timing_point.beat_length_ms / 4.0
        beat_position = (time_ms - timing_point.time_ms) / grid_spacing_ms
        snapped_position = round(beat_position)
        snapped_time_ms = timing_point.time_ms + snapped_position * grid_spacing_ms
        return int(round(snapped_time_ms))

    def _active_timing_point(
        self, control_points: list[TimingPoint], time_ms: float
    ) -> TimingPoint:
        active = control_points[0]
        for point in control_points:
            if point.time_ms <= time_ms:
                active = point
            else:
                break
        return active

    def _frame_index(self, snapped_time_ms: int, features: ExtractedFeatures) -> int:
        return min(
            features.cqt_magnitude.shape[1] - 1,
            max(
                0,
                int(
                    round(
                        (snapped_time_ms / 1000)
                        * features.sample_rate
                        / self.config.hop_length
                    )
                ),
            ),
        )

    def _lane_energy_matrix(self, cqt_magnitude: np.ndarray) -> np.ndarray:
        chunks = np.array_split(cqt_magnitude, self.key_count, axis=0)
        lane_energy = np.vstack([np.sum(chunk, axis=0) for chunk in chunks])
        return lane_energy

    def _smoothed_lane_scores(
        self, lane_energy: np.ndarray, center_frame_index: int
    ) -> np.ndarray:
        start = max(0, center_frame_index - self.smooth_frames)
        end = min(lane_energy.shape[1], center_frame_index + self.smooth_frames + 1)
        window = lane_energy[:, start:end]
        if window.shape[1] == 0:
            return lane_energy[:, center_frame_index]
        return np.mean(window, axis=1)

    def _lane_scores(self, lane_vector: np.ndarray) -> list[tuple[int, float]]:
        scores = []
        for lane, score in enumerate(lane_vector):
            scores.append((lane, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def _absolute_lane_threshold(self, lane_energy: np.ndarray) -> float:
        flattened = lane_energy.reshape(-1)
        percentile_value = float(
            np.percentile(flattened, self.chord_absolute_percentile)
        )
        return percentile_value * self.chord_absolute_threshold_ratio

    def _select_lanes(
        self,
        snapped_time_ms: int,
        lane_scores: list[tuple[int, float]],
        used_lanes_per_time: dict[int, set[int]],
        absolute_lane_threshold: float,
    ) -> list[int]:
        occupied = used_lanes_per_time.get(snapped_time_ms, set())
        selected: list[int] = []
        top_score = lane_scores[0][1] if lane_scores else 0.0

        for lane, score in lane_scores:
            if lane not in occupied:
                selected.append(lane)
                top_score = score
                break

        if not selected:
            return []

        if len(lane_scores) > 1 and top_score > 0:
            for lane, score in lane_scores:
                if lane in occupied or lane == selected[0]:
                    continue
                if (
                    score >= top_score * self.chord_relative_threshold
                    or score >= absolute_lane_threshold
                ):
                    selected.append(lane)
                if len(selected) >= self.max_chord_size:
                    break

        return selected

    def _estimate_hold_end_time_ms(
        self,
        start_time_ms: int,
        lane: int,
        start_frame_index: int,
        lane_energy: np.ndarray,
        local_peak_score: float,
        features: ExtractedFeatures,
        control_points: list[TimingPoint],
    ) -> int | None:
        hold_threshold = max(local_peak_score * self.hold_relative_threshold, 0.0)
        end_frame_index = start_frame_index
        consecutive_frames = 0

        for frame_index in range(start_frame_index + 1, lane_energy.shape[1]):
            score = float(lane_energy[lane, frame_index])
            window_start = max(0, frame_index - self.smooth_frames)
            window_end = min(lane_energy.shape[1], frame_index + self.smooth_frames + 1)
            window = lane_energy[lane, window_start:window_end]
            mean_value = float(np.mean(window)) if len(window) else 0.0
            if mean_value <= 0:
                stability = 0.0
            else:
                stability = float(np.std(window) / mean_value)

            if score >= hold_threshold and stability <= self.hold_stability_threshold:
                consecutive_frames += 1
                end_frame_index = frame_index
            else:
                if consecutive_frames > 0:
                    break

        if end_frame_index <= start_frame_index:
            return None

        duration_ms = (
            (end_frame_index - start_frame_index)
            * self.config.hop_length
            / features.sample_rate
            * 1000
        )
        if duration_ms < self.hold_min_duration_ms:
            return None

        raw_end_time_ms = start_time_ms + duration_ms
        snapped_end_time_ms = self._snap_time_ms(raw_end_time_ms, control_points)
        if snapped_end_time_ms <= start_time_ms:
            return None
        return snapped_end_time_ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于当前特征提取的规则基线生成 mania .osu 谱面"
    )
    parser.add_argument("--osu", type=Path, required=True, help="参考 osu!mania 谱面")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BEATMAP_OUTPUT_DIR,
        help="输出目录，默认 output/beatmaps",
    )
    parser.add_argument(
        "--dense-onset", action="store_true", help="使用高密度 onset 预设"
    )
    parser.add_argument("--key-count", type=int, default=6, help="目标键数，默认 6K")
    parser.add_argument(
        "--difficulty-name", default="RuleBaseline", help="生成谱面的难度名"
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=1,
        help="CQT 轨道判定时向前后平滑的帧数，默认 1",
    )
    parser.add_argument(
        "--chord-relative-threshold",
        type=float,
        default=0.7,
        help="第二高轨能量相对于第一高轨的阈值，超过则生成多押，默认 0.7",
    )
    parser.add_argument(
        "--chord-absolute-threshold-ratio",
        type=float,
        default=0.35,
        help="基于全局轨道能量分布的绝对阈值比例，超过即允许生成额外多押，默认 0.35",
    )
    parser.add_argument(
        "--chord-absolute-percentile",
        type=float,
        default=75.0,
        help="计算绝对阈值时使用的全局分位数，默认 75",
    )
    parser.add_argument(
        "--max-chord-size",
        type=int,
        default=3,
        help="单个时间点最多生成多少个按键，默认 3",
    )
    parser.add_argument(
        "--hold-min-duration-ms",
        type=int,
        default=180,
        help="长条最小时长，默认 180ms",
    )
    parser.add_argument(
        "--hold-relative-threshold",
        type=float,
        default=0.45,
        help="长条延伸时相对起始峰值的阈值比例，默认 0.45",
    )
    parser.add_argument(
        "--hold-stability-threshold",
        type=float,
        default=0.60,
        help="长条延伸时允许的能量波动系数上限，默认 0.60",
    )
    parser.add_argument("--onset-delta", type=float)
    parser.add_argument("--onset-wait", type=int)
    parser.add_argument("--onset-pre-max", type=int)
    parser.add_argument("--onset-post-max", type=int)
    parser.add_argument("--onset-pre-avg", type=int)
    parser.add_argument("--onset-post-avg", type=int)
    parser.add_argument("--onset-aggregate", choices=["mean", "median"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    osu_data = parse_osu_file(args.osu)
    audio_path = args.osu.parent / osu_data.general["AudioFilename"]

    config = dense_onset_config() if args.dense_onset else FeatureConfig()
    config = _apply_onset_overrides(config, args)
    features = extract_features(audio_path, config=config)

    mapper = RuleBasedMapper(
        config=config,
        key_count=args.key_count,
        smooth_frames=args.smooth_frames,
        chord_relative_threshold=args.chord_relative_threshold,
        chord_absolute_threshold_ratio=args.chord_absolute_threshold_ratio,
        chord_absolute_percentile=args.chord_absolute_percentile,
        max_chord_size=args.max_chord_size,
        hold_min_duration_ms=args.hold_min_duration_ms,
        hold_relative_threshold=args.hold_relative_threshold,
        hold_stability_threshold=args.hold_stability_threshold,
    )
    notes = mapper.generate_notes(osu_data, features)

    sample_dir = args.output_dir / _safe_stem(args.osu)
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{_safe_stem(args.osu)} [{args.difficulty_name}].osu"

    writer = ManiaBeatmapWriter()
    writer.write_from_reference(
        output_path=output_path,
        reference=osu_data,
        notes=notes,
        audio_filename=osu_data.general["AudioFilename"],
        key_count=args.key_count,
        version_name=args.difficulty_name,
    )

    print(f"Generated beatmap: {output_path}")
    print(f"Generated notes: {len(notes)}")
    print(
        f"Mapping params: key_count={args.key_count} smooth_frames={args.smooth_frames} chord_relative_threshold={args.chord_relative_threshold} chord_absolute_threshold_ratio={args.chord_absolute_threshold_ratio} max_chord_size={args.max_chord_size} hold_min_duration_ms={args.hold_min_duration_ms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

automakeosufile/config.py
```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FeatureConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmax: int = 8000
    cqt_bins_per_octave: int = 12
    cqt_n_bins: int = 84
    onset_backtrack: bool = False
    onset_aggregate: str = "median"
    onset_pre_max: int | None = None
    onset_post_max: int | None = None
    onset_pre_avg: int | None = None
    onset_post_avg: int | None = None
    onset_wait: int | None = None
    onset_delta: float | None = None
    songs_root: Path = Path(r"D:/osu!/Songs")
    onset_tolerance_ms: float = 50.0
    grid_divisors: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    visualize_window_seconds: float = 10.0
    max_visualize_notes: int | None = None


def dense_onset_config(songs_root: Path | None = None) -> FeatureConfig:
    base = FeatureConfig()
    return FeatureConfig(
        sample_rate=base.sample_rate,
        n_fft=base.n_fft,
        hop_length=base.hop_length,
        n_mels=base.n_mels,
        fmax=base.fmax,
        cqt_bins_per_octave=base.cqt_bins_per_octave,
        cqt_n_bins=base.cqt_n_bins,
        onset_backtrack=False,
        onset_aggregate="mean",
        onset_pre_max=1,
        onset_post_max=1,
        onset_pre_avg=1,
        onset_post_avg=1,
        onset_wait=1,
        onset_delta=0.02,
        songs_root=songs_root or base.songs_root,
        onset_tolerance_ms=base.onset_tolerance_ms,
        grid_divisors=base.grid_divisors,
        visualize_window_seconds=base.visualize_window_seconds,
        max_visualize_notes=base.max_visualize_notes,
    )
```

---

automakeosufile/dataset/__init__.py
```python
from .song_index import SongEntry, build_song_index, choose_sample_entry

__all__ = ["SongEntry", "build_song_index", "choose_sample_entry"]
```

---

automakeosufile/dataset/song_index.py
```python
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
```

---

automakeosufile/evaluation/__init__.py
```python
from .audio import (
    save_click_track_mix,
    save_predicted_click_track,
    save_reference_click_track,
)
from .metrics import (
    EventMatchMetrics,
    GridMetrics,
    HoldMetrics,
    LaneMetrics,
    compute_grid_metrics,
    compute_hold_metrics,
    compute_lane_metrics,
    match_note_times,
)
from .visualization import save_chroma_overlay_plot, save_onset_overlay_plot

__all__ = [
    "EventMatchMetrics",
    "GridMetrics",
    "HoldMetrics",
    "LaneMetrics",
    "compute_grid_metrics",
    "compute_hold_metrics",
    "compute_lane_metrics",
    "match_note_times",
    "save_click_track_mix",
    "save_predicted_click_track",
    "save_reference_click_track",
    "save_chroma_overlay_plot",
    "save_onset_overlay_plot",
]
```

---

automakeosufile/evaluation/audio.py
```python
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def save_click_track_mix(
    output_path: str | Path,
    audio_samples: np.ndarray,
    sample_rate: int,
    click_times_sec: list[float] | np.ndarray,
    click_freq: float,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clicks = librosa.clicks(
        times=np.asarray(click_times_sec, dtype=float),
        sr=sample_rate,
        click_freq=click_freq,
        click_duration=0.03,
        length=len(audio_samples),
    )

    mix = 0.7 * np.asarray(audio_samples, dtype=float) + 0.35 * clicks
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak

    sf.write(output_path, mix, sample_rate)
    return output_path


def save_reference_click_track(
    output_path: str | Path,
    audio_samples: np.ndarray,
    sample_rate: int,
    gt_times_sec: list[float] | np.ndarray,
) -> Path:
    return save_click_track_mix(
        output_path=output_path,
        audio_samples=audio_samples,
        sample_rate=sample_rate,
        click_times_sec=gt_times_sec,
        click_freq=880,
    )


def save_predicted_click_track(
    output_path: str | Path,
    audio_samples: np.ndarray,
    sample_rate: int,
    predicted_times_sec: list[float] | np.ndarray,
) -> Path:
    return save_click_track_mix(
        output_path=output_path,
        audio_samples=audio_samples,
        sample_rate=sample_rate,
        click_times_sec=predicted_times_sec,
        click_freq=1760,
    )
```

---

automakeosufile/evaluation/metrics.py
```python
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

from automakeosufile.parsers.osu_mania import OsuFileData, TimingPoint


@dataclass(slots=True)
class EventMatchMetrics:
    tolerance_ms: float
    matched_count: int
    predicted_count: int
    reference_count: int
    precision: float
    recall: float
    f1_score: float
    mean_abs_error_ms: float | None
    median_abs_error_ms: float | None
    max_abs_error_ms: float | None
    unmatched_predictions: int
    unmatched_references: int


@dataclass(slots=True)
class GridMetrics:
    note_count: int
    control_point_count: int
    median_error_ms: float | None
    mean_error_ms: float | None
    p95_error_ms: float | None
    max_error_ms: float | None
    dominant_divisor: int | None
    divisor_counts: dict[int, int]


@dataclass(slots=True)
class LaneMetrics:
    key_count: int
    lane_note_counts: dict[int, int]
    lane_hold_counts: dict[int, int]
    lane_balance_std: float | None
    lane_balance_cv: float | None
    chord_event_count: int
    chord_ratio: float
    max_chord_size: int
    mean_chord_size: float | None


@dataclass(slots=True)
class HoldMetrics:
    hold_count: int
    hold_ratio: float
    mean_duration_ms: float | None
    median_duration_ms: float | None
    p95_duration_ms: float | None
    max_duration_ms: float | None


def match_note_times(
    reference_times_ms: list[float] | np.ndarray,
    predicted_times_ms: list[float] | np.ndarray,
    tolerance_ms: float = 50.0,
) -> EventMatchMetrics:
    reference = sorted(float(value) for value in reference_times_ms)
    predicted = sorted(float(value) for value in predicted_times_ms)

    ref_index = 0
    pred_index = 0
    matched_errors: list[float] = []

    while ref_index < len(reference) and pred_index < len(predicted):
        ref_time = reference[ref_index]
        pred_time = predicted[pred_index]
        delta = pred_time - ref_time

        if abs(delta) <= tolerance_ms:
            matched_errors.append(abs(delta))
            ref_index += 1
            pred_index += 1
        elif pred_time < ref_time - tolerance_ms:
            pred_index += 1
        else:
            ref_index += 1

    matched_count = len(matched_errors)
    predicted_count = len(predicted)
    reference_count = len(reference)
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / reference_count if reference_count else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return EventMatchMetrics(
        tolerance_ms=tolerance_ms,
        matched_count=matched_count,
        predicted_count=predicted_count,
        reference_count=reference_count,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        mean_abs_error_ms=float(np.mean(matched_errors)) if matched_errors else None,
        median_abs_error_ms=float(median(matched_errors)) if matched_errors else None,
        max_abs_error_ms=float(max(matched_errors)) if matched_errors else None,
        unmatched_predictions=predicted_count - matched_count,
        unmatched_references=reference_count - matched_count,
    )


def compute_grid_metrics(
    osu_data: OsuFileData, divisors: tuple[int, ...]
) -> GridMetrics:
    control_points = osu_data.control_timing_points
    if not control_points or not osu_data.hit_objects:
        return GridMetrics(
            note_count=len(osu_data.hit_objects),
            control_point_count=len(control_points),
            median_error_ms=None,
            mean_error_ms=None,
            p95_error_ms=None,
            max_error_ms=None,
            dominant_divisor=None,
            divisor_counts={},
        )

    sorted_points = sorted(control_points, key=lambda point: point.time_ms)
    errors: list[float] = []
    divisor_counts: dict[int, int] = {divisor: 0 for divisor in divisors}

    for hit_object in osu_data.hit_objects:
        timing_point = _active_timing_point(sorted_points, hit_object.time_ms)
        if timing_point is None:
            continue

        error_ms, best_divisor = _best_grid_error_ms(
            note_time_ms=hit_object.time_ms,
            timing_point=timing_point,
            divisors=divisors,
        )
        errors.append(error_ms)
        divisor_counts[best_divisor] = divisor_counts.get(best_divisor, 0) + 1

    dominant_divisor = None
    if divisor_counts:
        dominant_divisor = max(
            divisor_counts, key=lambda divisor: divisor_counts[divisor]
        )

    return GridMetrics(
        note_count=len(osu_data.hit_objects),
        control_point_count=len(sorted_points),
        median_error_ms=float(np.median(errors)) if errors else None,
        mean_error_ms=float(np.mean(errors)) if errors else None,
        p95_error_ms=float(np.percentile(errors, 95)) if errors else None,
        max_error_ms=float(np.max(errors)) if errors else None,
        dominant_divisor=dominant_divisor,
        divisor_counts={k: v for k, v in divisor_counts.items() if v > 0},
    )


def compute_lane_metrics(osu_data: OsuFileData) -> LaneMetrics:
    lane_note_counts = {lane: 0 for lane in range(osu_data.key_count)}
    lane_hold_counts = {lane: 0 for lane in range(osu_data.key_count)}
    chord_sizes: list[int] = []
    grouped_by_time: dict[int, set[int]] = {}

    for hit_object in osu_data.hit_objects:
        lane_note_counts[hit_object.lane] = lane_note_counts.get(hit_object.lane, 0) + 1
        if hit_object.is_hold:
            lane_hold_counts[hit_object.lane] = (
                lane_hold_counts.get(hit_object.lane, 0) + 1
            )
        grouped_by_time.setdefault(hit_object.time_ms, set()).add(hit_object.lane)

    for lanes in grouped_by_time.values():
        chord_sizes.append(len(lanes))

    note_counts = np.asarray(list(lane_note_counts.values()), dtype=float)
    lane_balance_std = float(np.std(note_counts)) if len(note_counts) else None
    mean_count = float(np.mean(note_counts)) if len(note_counts) else 0.0
    lane_balance_cv = (lane_balance_std / mean_count) if mean_count > 0 else None
    chord_events = [size for size in chord_sizes if size > 1]

    return LaneMetrics(
        key_count=osu_data.key_count,
        lane_note_counts=lane_note_counts,
        lane_hold_counts=lane_hold_counts,
        lane_balance_std=lane_balance_std,
        lane_balance_cv=lane_balance_cv,
        chord_event_count=len(chord_events),
        chord_ratio=(len(chord_events) / len(chord_sizes)) if chord_sizes else 0.0,
        max_chord_size=max(chord_sizes) if chord_sizes else 0,
        mean_chord_size=float(np.mean(chord_sizes)) if chord_sizes else None,
    )


def compute_hold_metrics(osu_data: OsuFileData) -> HoldMetrics:
    hold_durations = [
        float(hit_object.end_time_ms - hit_object.time_ms)
        for hit_object in osu_data.hit_objects
        if hit_object.is_hold
        and hit_object.end_time_ms is not None
        and hit_object.end_time_ms >= hit_object.time_ms
    ]
    hold_count = len(hold_durations)
    note_count = len(osu_data.hit_objects)

    return HoldMetrics(
        hold_count=hold_count,
        hold_ratio=(hold_count / note_count) if note_count else 0.0,
        mean_duration_ms=float(np.mean(hold_durations)) if hold_durations else None,
        median_duration_ms=float(np.median(hold_durations)) if hold_durations else None,
        p95_duration_ms=(
            float(np.percentile(hold_durations, 95)) if hold_durations else None
        ),
        max_duration_ms=float(np.max(hold_durations)) if hold_durations else None,
    )


def _active_timing_point(
    control_points: list[TimingPoint], time_ms: int
) -> TimingPoint | None:
    active: TimingPoint | None = None
    for point in control_points:
        if point.time_ms <= time_ms:
            active = point
        else:
            break
    return active or (control_points[0] if control_points else None)


def _best_grid_error_ms(
    note_time_ms: int,
    timing_point: TimingPoint,
    divisors: tuple[int, ...],
) -> tuple[float, int]:
    best_error_ms = float("inf")
    best_divisor = divisors[0]

    for divisor in divisors:
        grid_spacing_ms = timing_point.beat_length_ms / divisor
        beat_position = (note_time_ms - timing_point.time_ms) / grid_spacing_ms
        snapped_position = round(beat_position)
        snapped_time = timing_point.time_ms + snapped_position * grid_spacing_ms
        error_ms = abs(note_time_ms - snapped_time)
        if error_ms < best_error_ms:
            best_error_ms = error_ms
            best_divisor = divisor

    return float(best_error_ms), best_divisor
```

---

automakeosufile/evaluation/visualization.py
```python
from __future__ import annotations

from pathlib import Path
from math import ceil

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from automakeosufile.config import FeatureConfig
from automakeosufile.features.extractor import ExtractedFeatures
from automakeosufile.parsers.osu_mania import OsuFileData


def save_onset_overlay_plot(
    output_path: str | Path,
    features: ExtractedFeatures,
    osu_data: OsuFileData,
    config: FeatureConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_objects = _limit_objects(osu_data, config)
    gt_times_sec = np.array([obj.time_ms / 1000 for obj in gt_objects])
    onset_times_sec = features.onset_times
    envelope_times = (
        np.arange(len(features.onset_envelope))
        * config.hop_length
        / features.sample_rate
    )
    lane_times = np.array([obj.time_ms / 1000 for obj in gt_objects])
    lane_values = np.array([obj.lane for obj in gt_objects])
    hold_mask = np.array([obj.is_hold for obj in gt_objects])

    window_seconds = config.visualize_window_seconds
    segment_count = max(1, ceil(features.duration_seconds / window_seconds))
    fig, axes = plt.subplots(
        nrows=segment_count * 2,
        ncols=1,
        figsize=(18, max(6, segment_count * 3.5)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)

    envelope_peak = (
        float(np.max(features.onset_envelope)) if len(features.onset_envelope) else 1.0
    )

    for segment_index in range(segment_count):
        start_sec = segment_index * window_seconds
        end_sec = min(features.duration_seconds, start_sec + window_seconds)
        envelope_ax = axes[segment_index * 2]
        lane_ax = axes[segment_index * 2 + 1]

        envelope_mask = (envelope_times >= start_sec) & (envelope_times < end_sec)
        gt_mask = (gt_times_sec >= start_sec) & (gt_times_sec < end_sec)
        onset_mask = (onset_times_sec >= start_sec) & (onset_times_sec < end_sec)
        lane_mask = (lane_times >= start_sec) & (lane_times < end_sec)

        envelope_ax.plot(
            envelope_times[envelope_mask],
            features.onset_envelope[envelope_mask],
            color="cyan",
            linewidth=1.0,
        )
        envelope_ax.vlines(
            gt_times_sec[gt_mask],
            ymin=0,
            ymax=envelope_peak,
            colors="orange",
            alpha=0.3,
            linewidth=0.8,
        )
        envelope_ax.vlines(
            onset_times_sec[onset_mask],
            ymin=0,
            ymax=envelope_peak,
            colors="lime",
            alpha=0.35,
            linewidth=0.8,
        )
        envelope_ax.set_xlim(start_sec, end_sec)
        envelope_ax.set_ylabel("Onset")
        envelope_ax.set_title(f"Onset overlay  {start_sec:.0f}s - {end_sec:.0f}s")

        lane_ax.scatter(
            lane_times[lane_mask],
            lane_values[lane_mask],
            s=12,
            c=np.where(hold_mask[lane_mask], "tomato", "gold"),
            alpha=0.9,
            edgecolors="none",
        )
        lane_ax.set_xlim(start_sec, end_sec)
        lane_ax.set_ylabel("Lane")
        lane_ax.set_xlabel("Time (s)")
        lane_ax.set_ylim(-0.5, max(0.5, osu_data.key_count - 0.5))
        lane_ax.set_title("GT note raster (tap=gold, hold=red)")
        lane_ax.set_facecolor("#111111")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_chroma_overlay_plot(
    output_path: str | Path,
    features: ExtractedFeatures,
    osu_data: OsuFileData,
    config: FeatureConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_objects = _limit_objects(osu_data, config)
    gt_times_sec = np.array([obj.time_ms / 1000 for obj in gt_objects])
    lane_values = np.array([obj.lane for obj in gt_objects])
    chroma_times = (
        np.arange(features.chroma_cqt.shape[1])
        * config.hop_length
        / features.sample_rate
    )
    window_seconds = config.visualize_window_seconds
    segment_count = max(1, ceil(features.duration_seconds / window_seconds))

    fig, axes = plt.subplots(
        nrows=segment_count * 2,
        ncols=1,
        figsize=(18, max(7, segment_count * 3.8)),
        sharex=False,
        height_ratios=[3.2, 1.0] * segment_count,
    )
    axes = np.atleast_1d(axes)

    for segment_index in range(segment_count):
        start_sec = segment_index * window_seconds
        end_sec = min(features.duration_seconds, start_sec + window_seconds)
        chroma_ax = axes[segment_index * 2]
        lane_ax = axes[segment_index * 2 + 1]

        frame_mask = (chroma_times >= start_sec) & (chroma_times < end_sec)
        gt_mask = (gt_times_sec >= start_sec) & (gt_times_sec < end_sec)

        if np.any(frame_mask):
            chroma_slice = features.chroma_cqt[:, frame_mask]
            chroma_ax.imshow(
                chroma_slice,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[start_sec, end_sec, 0, 12],
            )
        chroma_ax.vlines(
            gt_times_sec[gt_mask],
            ymin=0,
            ymax=11.5,
            colors="white",
            alpha=0.18,
            linewidth=0.7,
        )
        chroma_ax.set_xlim(start_sec, end_sec)
        chroma_ax.set_ylim(0, 12)
        chroma_ax.set_ylabel("Chroma")
        chroma_ax.set_title(f"Chroma overlay  {start_sec:.0f}s - {end_sec:.0f}s")

        lane_ax.scatter(
            gt_times_sec[gt_mask],
            lane_values[gt_mask],
            s=12,
            c="gold",
            alpha=0.9,
            edgecolors="none",
        )
        lane_ax.set_xlim(start_sec, end_sec)
        lane_ax.set_ylim(-0.5, max(0.5, osu_data.key_count - 0.5))
        lane_ax.set_ylabel("Lane")
        lane_ax.set_xlabel("Time (s)")
        lane_ax.set_title("GT note lanes")
        lane_ax.set_facecolor("#111111")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _limit_objects(osu_data: OsuFileData, config: FeatureConfig):
    if config.max_visualize_notes is None:
        return osu_data.hit_objects
    return osu_data.hit_objects[: config.max_visualize_notes]
```

---

automakeosufile/features/__init__.py
```python
from .extractor import ExtractedFeatures, extract_features

__all__ = ["ExtractedFeatures", "extract_features"]
```

---

automakeosufile/features/extractor.py
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from automakeosufile.config import FeatureConfig


@dataclass(slots=True)
class ExtractedFeatures:
    audio_path: Path
    audio_samples: np.ndarray
    duration_seconds: float
    sample_rate: int
    stft_complex: np.ndarray
    stft_magnitude: np.ndarray
    stft_db: np.ndarray
    mel_power: np.ndarray
    mel_db: np.ndarray
    rms: np.ndarray
    onset_envelope: np.ndarray
    onset_frames: np.ndarray
    onset_times: np.ndarray
    cqt_magnitude: np.ndarray
    chroma_cqt: np.ndarray


def extract_features(
    audio_path: str | Path, config: FeatureConfig | None = None
) -> ExtractedFeatures:
    config = config or FeatureConfig()
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=config.sample_rate)
    aggregate = _get_onset_aggregate(config.onset_aggregate)

    stft_complex = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    stft_magnitude = np.abs(stft_complex)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmax=config.fmax,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    rms = librosa.feature.rms(
        S=stft_magnitude, frame_length=config.n_fft, hop_length=config.hop_length
    )[0]

    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=config.hop_length,
        aggregate=aggregate,
    )
    onset_detect_kwargs = {
        "onset_envelope": onset_envelope,
        "sr": sr,
        "hop_length": config.hop_length,
        "backtrack": config.onset_backtrack,
    }
    if config.onset_pre_max is not None:
        onset_detect_kwargs["pre_max"] = config.onset_pre_max
    if config.onset_post_max is not None:
        onset_detect_kwargs["post_max"] = config.onset_post_max
    if config.onset_pre_avg is not None:
        onset_detect_kwargs["pre_avg"] = config.onset_pre_avg
    if config.onset_post_avg is not None:
        onset_detect_kwargs["post_avg"] = config.onset_post_avg
    if config.onset_wait is not None:
        onset_detect_kwargs["wait"] = config.onset_wait
    if config.onset_delta is not None:
        onset_detect_kwargs["delta"] = config.onset_delta

    onset_frames = librosa.onset.onset_detect(**onset_detect_kwargs)
    onset_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=config.hop_length
    )

    cqt_magnitude = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=config.hop_length,
            bins_per_octave=config.cqt_bins_per_octave,
            n_bins=config.cqt_n_bins,
        )
    )
    chroma_cqt = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=config.hop_length,
        bins_per_octave=config.cqt_bins_per_octave,
    )

    return ExtractedFeatures(
        audio_path=audio_path,
        audio_samples=y,
        duration_seconds=float(librosa.get_duration(y=y, sr=sr)),
        sample_rate=int(sr),
        stft_complex=stft_complex,
        stft_magnitude=stft_magnitude,
        stft_db=stft_db,
        mel_power=mel_power,
        mel_db=mel_db,
        rms=rms,
        onset_envelope=onset_envelope,
        onset_frames=onset_frames,
        onset_times=onset_times,
        cqt_magnitude=cqt_magnitude,
        chroma_cqt=chroma_cqt,
    )


def _get_onset_aggregate(name: str):
    normalized = name.strip().lower()
    if normalized == "mean":
        return np.mean
    return np.median
```

---

automakeosufile/io/__init__.py
```python
from .mania_writer import ManiaBeatmapWriter, ManiaNote

__all__ = ["ManiaBeatmapWriter", "ManiaNote"]
```

---

automakeosufile/io/mania_writer.py
```python
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
```

---

automakeosufile/output_paths.py
```python
from __future__ import annotations

from pathlib import Path


OUTPUT_ROOT = Path("output")
INSPECT_OUTPUT_DIR = OUTPUT_ROOT / "inspect"
OPTIMIZE_OUTPUT_DIR = OUTPUT_ROOT / "optimize"
BEATMAP_OUTPUT_DIR = OUTPUT_ROOT / "beatmaps" / "current"
```

---

automakeosufile/parsers/__init__.py
```python
from .osu_mania import ManiaHitObject, OsuFileData, TimingPoint, parse_osu_file

__all__ = ["ManiaHitObject", "OsuFileData", "TimingPoint", "parse_osu_file"]
```

---

automakeosufile/parsers/osu_mania.py
```python
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
```

---

automakeosufile/tools/__init__.py
```python
"""新主流程下的命令行工具入口。"""
```

---

automakeosufile/tools/inspect_sample.py
```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from automakeosufile.config import FeatureConfig, dense_onset_config
from automakeosufile.dataset.song_index import build_song_index, choose_sample_entry
from automakeosufile.evaluation.audio import (
    save_predicted_click_track,
    save_reference_click_track,
)
from automakeosufile.evaluation.metrics import (
    compute_grid_metrics,
    compute_hold_metrics,
    compute_lane_metrics,
    match_note_times,
)
from automakeosufile.evaluation.visualization import (
    save_chroma_overlay_plot,
    save_onset_overlay_plot,
)
from automakeosufile.features.extractor import extract_features
from automakeosufile.output_paths import INSPECT_OUTPUT_DIR
from automakeosufile.parsers.osu_mania import parse_osu_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查一首 osu!mania 样本的音频与谱面对齐信息"
    )
    parser.add_argument("--songs-root", type=Path, default=FeatureConfig().songs_root)
    parser.add_argument("--osu", type=Path, help="直接指定一个 .osu 文件")
    parser.add_argument(
        "--audio",
        type=Path,
        help="可选：显式指定音频路径，适用于 output 下生成谱面的质检",
    )
    parser.add_argument(
        "--dense-onset",
        action="store_true",
        help="使用更激进的高密度 onset 参数预设",
    )
    parser.add_argument("--onset-delta", type=float, help="覆盖 onset delta 阈值")
    parser.add_argument("--onset-wait", type=int, help="覆盖 onset wait 参数")
    parser.add_argument("--onset-pre-max", type=int, help="覆盖 onset pre_max 参数")
    parser.add_argument("--onset-post-max", type=int, help="覆盖 onset post_max 参数")
    parser.add_argument("--onset-pre-avg", type=int, help="覆盖 onset pre_avg 参数")
    parser.add_argument("--onset-post-avg", type=int, help="覆盖 onset post_avg 参数")
    parser.add_argument(
        "--onset-aggregate",
        choices=["mean", "median"],
        help="覆盖 onset_strength 的聚合方式",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INSPECT_OUTPUT_DIR,
        help="输出质检图与指标 JSON 的目录，默认 output/inspect",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.osu:
        parsed = parse_osu_file(args.osu)
        audio_path = args.audio or (args.osu.parent / parsed.general["AudioFilename"])
    else:
        entries = build_song_index(args.songs_root)
        entry = choose_sample_entry(entries)
        parsed = parse_osu_file(entry.osu_path)
        audio_path = args.audio or entry.audio_path

    config = (
        dense_onset_config(args.songs_root)
        if args.dense_onset
        else FeatureConfig(songs_root=args.songs_root)
    )
    config = _apply_onset_overrides(config, args)
    features = extract_features(audio_path, config=config)
    hold_count = sum(1 for obj in parsed.hit_objects if obj.is_hold)
    gt_note_times_ms = [obj.time_ms for obj in parsed.hit_objects]
    predicted_onset_times_ms = list(features.onset_times * 1000)

    onset_metrics = match_note_times(
        reference_times_ms=gt_note_times_ms,
        predicted_times_ms=predicted_onset_times_ms,
        tolerance_ms=config.onset_tolerance_ms,
    )
    grid_metrics = compute_grid_metrics(parsed, config.grid_divisors)
    lane_metrics = compute_lane_metrics(parsed)
    hold_metrics = compute_hold_metrics(parsed)

    print(f"OSU: {parsed.path}")
    print(f"Audio: {audio_path}")
    print(f"Mode: {parsed.mode}  Keys: {parsed.key_count}")
    print(
        f"Title: {parsed.metadata.get('Artist', '')} - {parsed.metadata.get('Title', '')} [{parsed.metadata.get('Version', '')}]"
    )
    print(f"HitObjects: {len(parsed.hit_objects)}  Holds: {hold_count}")
    print(f"TimingPoints: {len(parsed.timing_points)}")
    print(f"Duration: {features.duration_seconds:.2f}s")
    print(f"Onsets: {len(features.onset_times)}")
    print(
        f"Mel shape: {features.mel_db.shape}  Chroma shape: {features.chroma_cqt.shape}"
    )
    print(
        f"Onset params: aggregate={config.onset_aggregate} delta={_fmt(config.onset_delta)} wait={_fmt_int(config.onset_wait)} pre_max={_fmt_int(config.onset_pre_max)} post_max={_fmt_int(config.onset_post_max)} pre_avg={_fmt_int(config.onset_pre_avg)} post_avg={_fmt_int(config.onset_post_avg)}"
    )

    print(
        f"Onset metrics: P={onset_metrics.precision:.3f} R={onset_metrics.recall:.3f} F1={onset_metrics.f1_score:.3f}"
    )
    print(
        f"Onset timing error (matched): mean={_fmt(onset_metrics.mean_abs_error_ms)} ms median={_fmt(onset_metrics.median_abs_error_ms)} ms"
    )
    print(
        f"Grid error: median={_fmt(grid_metrics.median_error_ms)} ms p95={_fmt(grid_metrics.p95_error_ms)} ms dominant_divisor={grid_metrics.dominant_divisor}"
    )
    print(
        f"Lane metrics: balance_cv={_fmt_ratio(lane_metrics.lane_balance_cv)} chord_ratio={lane_metrics.chord_ratio:.3f} max_chord={lane_metrics.max_chord_size}"
    )
    print(
        f"Hold metrics: ratio={hold_metrics.hold_ratio:.3f} mean={_fmt(hold_metrics.mean_duration_ms)} ms p95={_fmt(hold_metrics.p95_duration_ms)} ms"
    )

    sample_dir = args.output_dir / _safe_stem(parsed.path)
    sample_dir.mkdir(parents=True, exist_ok=True)

    onset_plot = save_onset_overlay_plot(
        sample_dir / "onset_overlay.png",
        features=features,
        osu_data=parsed,
        config=config,
    )
    chroma_plot = save_chroma_overlay_plot(
        sample_dir / "chroma_overlay.png",
        features=features,
        osu_data=parsed,
        config=config,
    )
    click_track_gt = save_reference_click_track(
        sample_dir / "click_track_gt.wav",
        audio_samples=features.audio_samples,
        sample_rate=features.sample_rate,
        gt_times_sec=[value / 1000 for value in gt_note_times_ms],
    )
    click_track_pred = save_predicted_click_track(
        sample_dir / "click_track_pred.wav",
        audio_samples=features.audio_samples,
        sample_rate=features.sample_rate,
        predicted_times_sec=list(features.onset_times),
    )

    metrics_payload = {
        "osu_path": str(parsed.path),
        "audio_path": str(audio_path),
        "mode": parsed.mode,
        "key_count": parsed.key_count,
        "note_count": len(parsed.hit_objects),
        "hold_count": hold_count,
        "feature_config": _serialize_config(config),
        "onset_metrics": asdict(onset_metrics),
        "grid_metrics": asdict(grid_metrics),
        "lane_metrics": asdict(lane_metrics),
        "hold_metrics": asdict(hold_metrics),
        "plots": {
            "onset_overlay": str(onset_plot),
            "chroma_overlay": str(chroma_plot),
        },
        "audio": {
            "click_track_gt": str(click_track_gt),
            "click_track_pred": str(click_track_pred),
        },
    }
    metrics_path = sample_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved outputs to: {sample_dir}")

    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _serialize_config(config: FeatureConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized["songs_root"] = str(config.songs_root)
    serialized["grid_divisors"] = list(config.grid_divisors)
    return serialized


def _apply_onset_overrides(config: FeatureConfig, args) -> FeatureConfig:
    if args.onset_delta is not None:
        config.onset_delta = args.onset_delta
    if args.onset_wait is not None:
        config.onset_wait = args.onset_wait
    if args.onset_pre_max is not None:
        config.onset_pre_max = args.onset_pre_max
    if args.onset_post_max is not None:
        config.onset_post_max = args.onset_post_max
    if args.onset_pre_avg is not None:
        config.onset_pre_avg = args.onset_pre_avg
    if args.onset_post_avg is not None:
        config.onset_post_avg = args.onset_post_avg
    if args.onset_aggregate is not None:
        config.onset_aggregate = args.onset_aggregate
    return config


def _safe_stem(path: Path) -> str:
    name = path.stem
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
```

---

automakeosufile/tools/optimize_onset_params.py
```python
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from automakeosufile.config import FeatureConfig, dense_onset_config
from automakeosufile.evaluation.metrics import EventMatchMetrics, match_note_times
from automakeosufile.output_paths import OPTIMIZE_OUTPUT_DIR
from automakeosufile.parsers.osu_mania import parse_osu_file


@dataclass(slots=True)
class OnsetParamSet:
    onset_aggregate: str
    onset_delta: float | None
    onset_wait: int | None
    onset_pre_max: int | None
    onset_post_max: int | None
    onset_pre_avg: int | None
    onset_post_avg: int | None


@dataclass(slots=True)
class OptimizationResult:
    params: OnsetParamSet
    metrics: EventMatchMetrics
    onset_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用网格搜索自动寻找 onset 参数的最佳组合"
    )
    parser.add_argument("--osu", type=Path, required=True, help="参考 osu!mania 谱面")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OPTIMIZE_OUTPUT_DIR,
        help="输出 best_params.json 与全量搜索结果的目录，默认 output/optimize",
    )
    parser.add_argument(
        "--dense-onset",
        action="store_true",
        help="以高密度预设作为基础配置",
    )
    parser.add_argument(
        "--aggregate-values",
        default="mean,median",
        help="候选聚合方式，逗号分隔，例如 mean,median",
    )
    parser.add_argument(
        "--delta-values",
        default="0.01,0.02,0.03,0.05,0.07",
        help="候选 delta，逗号分隔",
    )
    parser.add_argument(
        "--wait-values",
        default="1,2,3",
        help="候选 wait，逗号分隔",
    )
    parser.add_argument(
        "--pre-max-values",
        default="1,2,3",
        help="候选 pre_max，逗号分隔",
    )
    parser.add_argument(
        "--post-max-values",
        default="1,2,3",
        help="候选 post_max，逗号分隔",
    )
    parser.add_argument(
        "--pre-avg-values",
        default="1,2,3",
        help="候选 pre_avg，逗号分隔",
    )
    parser.add_argument(
        "--post-avg-values",
        default="1,2,3",
        help="候选 post_avg，逗号分隔",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="输出前多少组结果到 top_results.json",
    )
    parser.add_argument(
        "--limit-trials",
        type=int,
        help="可选：只跑前 N 个组合，用于快速试跑",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    osu_data = parse_osu_file(args.osu)
    audio_path = args.osu.parent / osu_data.general["AudioFilename"]

    base_config = dense_onset_config() if args.dense_onset else FeatureConfig()
    y, sr = librosa.load(audio_path, sr=base_config.sample_rate)
    reference_times_ms = [obj.time_ms for obj in osu_data.hit_objects]

    aggregate_values = _parse_str_values(args.aggregate_values)
    delta_values = _parse_optional_float_values(args.delta_values)
    wait_values = _parse_optional_int_values(args.wait_values)
    pre_max_values = _parse_optional_int_values(args.pre_max_values)
    post_max_values = _parse_optional_int_values(args.post_max_values)
    pre_avg_values = _parse_optional_int_values(args.pre_avg_values)
    post_avg_values = _parse_optional_int_values(args.post_avg_values)

    search_space = list(
        itertools.product(
            aggregate_values,
            delta_values,
            wait_values,
            pre_max_values,
            post_max_values,
            pre_avg_values,
            post_avg_values,
        )
    )
    if args.limit_trials is not None:
        search_space = search_space[: args.limit_trials]

    print(f"Audio: {audio_path}")
    print(f"Reference notes: {len(reference_times_ms)}")
    print(f"Search trials: {len(search_space)}")

    envelope_cache: dict[str, np.ndarray] = {}
    results: list[OptimizationResult] = []

    for index, combination in enumerate(search_space, start=1):
        param_set = OnsetParamSet(
            onset_aggregate=combination[0],
            onset_delta=combination[1],
            onset_wait=combination[2],
            onset_pre_max=combination[3],
            onset_post_max=combination[4],
            onset_pre_avg=combination[5],
            onset_post_avg=combination[6],
        )

        envelope = envelope_cache.get(param_set.onset_aggregate)
        if envelope is None:
            envelope = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=base_config.hop_length,
                aggregate=np.mean if param_set.onset_aggregate == "mean" else np.median,
            )
            envelope_cache[param_set.onset_aggregate] = envelope

        onset_times_ms = _detect_onsets_ms(
            onset_envelope=envelope,
            sr=sr,
            hop_length=base_config.hop_length,
            params=param_set,
            backtrack=base_config.onset_backtrack,
        )
        metrics = match_note_times(
            reference_times_ms=reference_times_ms,
            predicted_times_ms=onset_times_ms,
            tolerance_ms=base_config.onset_tolerance_ms,
        )
        results.append(
            OptimizationResult(
                params=param_set,
                metrics=metrics,
                onset_count=len(onset_times_ms),
            )
        )

        if index == 1 or index % 25 == 0 or index == len(search_space):
            print(
                f"[{index}/{len(search_space)}] F1={metrics.f1_score:.3f} P={metrics.precision:.3f} R={metrics.recall:.3f} onsets={len(onset_times_ms)} params={_short_param_summary(param_set)}"
            )

    results.sort(
        key=lambda item: (
            item.metrics.f1_score,
            item.metrics.recall,
            item.metrics.precision,
            -abs(item.onset_count - len(reference_times_ms)),
        ),
        reverse=True,
    )
    best = results[0]

    output_dir = args.output_dir / _safe_stem(args.osu)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_payload = {
        "osu_path": str(args.osu),
        "audio_path": str(audio_path),
        "search_trial_count": len(search_space),
        "base_config": _serialize_base_config(base_config),
        "best": _serialize_result(best),
        "rerun_command": _build_rerun_command(args.osu, best.params),
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_payload = {
        "results": [_serialize_result(result) for result in results[: args.top_k]]
    }
    (output_dir / "top_results.json").write_text(
        json.dumps(top_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Best result:")
    print(
        f"F1={best.metrics.f1_score:.3f} P={best.metrics.precision:.3f} R={best.metrics.recall:.3f} onsets={best.onset_count}"
    )
    print(f"Params: {_short_param_summary(best.params)}")
    print(f"Saved: {output_dir}")
    print("Rerun with:")
    print(_build_rerun_command(args.osu, best.params))
    return 0


def _detect_onsets_ms(
    onset_envelope: np.ndarray,
    sr: int,
    hop_length: int,
    params: OnsetParamSet,
    backtrack: bool,
) -> list[float]:
    kwargs = {
        "onset_envelope": onset_envelope,
        "sr": sr,
        "hop_length": hop_length,
        "backtrack": backtrack,
    }
    if params.onset_delta is not None:
        kwargs["delta"] = params.onset_delta
    if params.onset_wait is not None:
        kwargs["wait"] = params.onset_wait
    if params.onset_pre_max is not None:
        kwargs["pre_max"] = params.onset_pre_max
    if params.onset_post_max is not None:
        kwargs["post_max"] = params.onset_post_max
    if params.onset_pre_avg is not None:
        kwargs["pre_avg"] = params.onset_pre_avg
    if params.onset_post_avg is not None:
        kwargs["post_avg"] = params.onset_post_avg

    onset_frames = librosa.onset.onset_detect(**kwargs)
    return list(
        librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) * 1000
    )


def _serialize_result(result: OptimizationResult) -> dict[str, object]:
    return {
        "params": asdict(result.params),
        "metrics": asdict(result.metrics),
        "onset_count": result.onset_count,
    }


def _serialize_base_config(config: FeatureConfig) -> dict[str, object]:
    return {
        "sample_rate": config.sample_rate,
        "hop_length": config.hop_length,
        "onset_tolerance_ms": config.onset_tolerance_ms,
        "onset_backtrack": config.onset_backtrack,
    }


def _build_rerun_command(osu_path: Path, params: OnsetParamSet) -> str:
    parts = [
        ".\\.venv\\Scripts\\python.exe -m automakeosufile.tools.inspect_sample",
        f'--osu "{osu_path}"',
        f"--onset-aggregate {params.onset_aggregate}",
    ]
    if params.onset_delta is not None:
        parts.append(f"--onset-delta {params.onset_delta}")
    if params.onset_wait is not None:
        parts.append(f"--onset-wait {params.onset_wait}")
    if params.onset_pre_max is not None:
        parts.append(f"--onset-pre-max {params.onset_pre_max}")
    if params.onset_post_max is not None:
        parts.append(f"--onset-post-max {params.onset_post_max}")
    if params.onset_pre_avg is not None:
        parts.append(f"--onset-pre-avg {params.onset_pre_avg}")
    if params.onset_post_avg is not None:
        parts.append(f"--onset-post-avg {params.onset_post_avg}")
    return " ".join(parts)


def _short_param_summary(params: OnsetParamSet) -> str:
    return (
        f"agg={params.onset_aggregate},delta={params.onset_delta},wait={params.onset_wait},"
        f"pre_max={params.onset_pre_max},post_max={params.onset_post_max},"
        f"pre_avg={params.onset_pre_avg},post_avg={params.onset_post_avg}"
    )


def _safe_stem(path: Path) -> str:
    name = path.stem
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name


def _parse_str_values(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("No aggregate values provided")
    return values


def _parse_optional_float_values(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "default"}:
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError("No float values provided")
    return values


def _parse_optional_int_values(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "default"}:
            values.append(None)
        else:
            values.append(int(token))
    if not values:
        raise ValueError("No int values provided")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
```

---

