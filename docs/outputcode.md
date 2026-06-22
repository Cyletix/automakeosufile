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

import matplotlib.pyplot as plt
import numpy as np

from automakeosufile.config import FeatureConfig, dense_onset_config
from automakeosufile.features.extractor import ExtractedFeatures, extract_features
from automakeosufile.output_paths import INSPECT_OUTPUT_DIR
from automakeosufile.parsers.osu_mania import OsuFileData, parse_osu_file
from automakeosufile.tools.inspect_sample import _apply_onset_overrides


@dataclass(slots=True)
class AlignmentResult:
    offset_frames: int
    offset_ms: float
    correlation_score: float
    aligned_times_ms: list[float]
    original_times_ms: list[float]
    searched_offsets_ms: list[float]
    searched_scores: list[float]


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
    searched_offsets_ms: list[float] = []
    searched_scores: list[float] = []

    for lag in range(-max_offset_frames, max_offset_frames + 1):
        score = _lagged_dot(onset_signal, reference_signal, lag)
        searched_offsets_ms.append(lag * hop_length / sample_rate * 1000.0)
        searched_scores.append(float(score))
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
        searched_offsets_ms=searched_offsets_ms,
        searched_scores=searched_scores,
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
    parser.add_argument("--onset-delta", type=float)
    parser.add_argument("--onset-wait", type=int)
    parser.add_argument("--onset-pre-max", type=int)
    parser.add_argument("--onset-post-max", type=int)
    parser.add_argument("--onset-pre-avg", type=int)
    parser.add_argument("--onset-post-avg", type=int)
    parser.add_argument("--onset-aggregate", choices=["mean", "median"])
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
    config = _apply_onset_overrides(config, args)
    result, osu_data, features = align_osu_to_audio(
        osu_path=args.osu,
        audio_path=args.audio,
        config=config,
        max_offset_ms=args.max_offset_ms,
    )

    output_dir = args.output_dir / _safe_stem(args.osu)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "offset_alignment.json"
    plot_path = output_dir / "offset_correlation.png"
    _save_correlation_plot(plot_path, result)
    payload = {
        "osu_path": str(args.osu),
        "audio_path": str(features.audio_path),
        "audio_filename": osu_data.general.get("AudioFilename", ""),
        "note_count": len(osu_data.hit_objects),
        "sample_rate": features.sample_rate,
        "hop_length": config.hop_length,
        "max_offset_ms": args.max_offset_ms,
        "feature_config": {
            "onset_aggregate": config.onset_aggregate,
            "onset_delta": config.onset_delta,
            "onset_wait": config.onset_wait,
            "onset_pre_max": config.onset_pre_max,
            "onset_post_max": config.onset_post_max,
            "onset_pre_avg": config.onset_pre_avg,
            "onset_post_avg": config.onset_post_avg,
        },
        "artifacts": {
            "offset_correlation_plot": str(plot_path),
        },
        "alignment": asdict(result),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Best offset: {result.offset_ms:.2f} ms ({result.offset_frames} frames)")
    print(f"Correlation score: {result.correlation_score:.4f}")
    print(f"Aligned note count: {len(result.aligned_times_ms)}")
    print(f"Saved correlation plot: {plot_path}")
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


def _save_correlation_plot(output_path: Path, result: AlignmentResult) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.searched_offsets_ms, result.searched_scores, color="#4ea1ff")
    ax.axvline(result.offset_ms, color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.scatter(
        [result.offset_ms], [result.correlation_score], color="#ff7f0e", zorder=3
    )
    ax.set_title("Offset cross-correlation curve")
    ax.set_xlabel("Offset (ms)")
    ax.set_ylabel("Correlation score")
    ax.grid(alpha=0.25)
    ax.text(
        result.offset_ms,
        result.correlation_score,
        f"  best={result.offset_ms:.2f} ms",
        color="#ff7f0e",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


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
        max_chord_size: int = 4,
        high_energy_percentile: float = 95.0,
        medium_energy_percentile: float = 80.0,
        high_energy_chord_size: int = 4,
        medium_energy_chord_size: int = 2,
        hold_min_duration_ms: int = 180,
        hold_relative_threshold: float = 0.45,
        hold_stability_threshold: float = 0.60,
    ):
        self.config = config
        self.key_count = key_count
        self.smooth_frames = max(0, smooth_frames)
        self.max_chord_size = max(1, max_chord_size)
        self.high_energy_percentile = high_energy_percentile
        self.medium_energy_percentile = medium_energy_percentile
        self.high_energy_chord_size = max(1, high_energy_chord_size)
        self.medium_energy_chord_size = max(1, medium_energy_chord_size)
        self.hold_min_duration_ms = hold_min_duration_ms
        self.hold_relative_threshold = hold_relative_threshold
        self.hold_stability_threshold = hold_stability_threshold
        self.lane_busy_until = {lane: 0 for lane in range(self.key_count)}

    def generate_notes(
        self, osu_data: OsuFileData, features: ExtractedFeatures
    ) -> list[ManiaNote]:
        control_points = sorted(
            osu_data.control_timing_points, key=lambda point: point.time_ms
        )
        if not control_points:
            raise ValueError("Reference beatmap has no uninherited timing points")

        raw_times_ms = [float(value) for value in features.onset_times * 1000]
        snapped_times_ms = sorted(
            set(self._snap_time_ms(value, control_points) for value in raw_times_ms)
        )
        next_onset_time_by_time = self._build_next_onset_time_lookup(snapped_times_ms)
        harmonic_cqt = features.cqt_magnitude
        lane_energy = self._lane_energy_matrix(harmonic_cqt)
        lane_energy_baseline = self._lane_energy_baseline(lane_energy)
        high_energy_threshold = float(
            np.percentile(features.rms, self.high_energy_percentile)
        )
        medium_energy_threshold = float(
            np.percentile(features.rms, self.medium_energy_percentile)
        )

        notes: list[ManiaNote] = []
        self.lane_busy_until = {lane: 0 for lane in range(self.key_count)}

        for snapped_time_ms in snapped_times_ms:
            frame_index = self._frame_index(snapped_time_ms, features)
            rms_frame_index = self._rms_frame_index(snapped_time_ms, features)
            max_notes_for_time = self._max_notes_for_time(
                rms_value=float(features.rms[rms_frame_index]),
                high_energy_threshold=high_energy_threshold,
                medium_energy_threshold=medium_energy_threshold,
            )
            available_lanes = self._available_lanes(snapped_time_ms)
            if not available_lanes:
                continue
            smoothed_lane_scores = self._smoothed_lane_scores(lane_energy, frame_index)
            lane_scores = self._lane_scores(
                smoothed_lane_scores=smoothed_lane_scores,
                lane_energy_baseline=lane_energy_baseline,
                available_lanes=available_lanes,
            )
            selected_lanes = self._select_lanes(
                lane_scores=lane_scores,
                max_notes_for_time=max_notes_for_time,
            )

            for lane, _score in selected_lanes:
                hold_end_time_ms = self._estimate_hold_end_time_ms(
                    start_time_ms=snapped_time_ms,
                    next_onset_time_ms=next_onset_time_by_time.get(snapped_time_ms),
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
                self.lane_busy_until[lane] = hold_end_time_ms or snapped_time_ms

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

    def _rms_frame_index(
        self, snapped_time_ms: int, features: ExtractedFeatures
    ) -> int:
        return min(
            len(features.rms) - 1,
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

    def _lane_energy_baseline(self, lane_energy: np.ndarray) -> np.ndarray:
        baseline = np.mean(lane_energy, axis=1)
        return np.maximum(baseline, 1e-6)

    def _build_next_onset_time_lookup(
        self, snapped_times_ms: list[int]
    ) -> dict[int, int | None]:
        unique_times = sorted(set(snapped_times_ms))
        lookup: dict[int, int | None] = {}
        for index, time_ms in enumerate(unique_times):
            next_time_ms = (
                unique_times[index + 1] if index + 1 < len(unique_times) else None
            )
            lookup[time_ms] = next_time_ms
        return lookup

    def _smoothed_lane_scores(
        self, lane_energy: np.ndarray, center_frame_index: int
    ) -> np.ndarray:
        start = max(0, center_frame_index - self.smooth_frames)
        end = min(lane_energy.shape[1], center_frame_index + self.smooth_frames + 1)
        window = lane_energy[:, start:end]
        if window.shape[1] == 0:
            return lane_energy[:, center_frame_index]
        return np.mean(window, axis=1)

    def _available_lanes(self, snapped_time_ms: int) -> list[int]:
        return [
            lane
            for lane in range(self.key_count)
            if snapped_time_ms >= self.lane_busy_until.get(lane, 0)
        ]

    def _lane_scores(
        self,
        smoothed_lane_scores: np.ndarray,
        lane_energy_baseline: np.ndarray,
        available_lanes: list[int],
    ) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for lane in available_lanes:
            whitened_score = float(
                smoothed_lane_scores[lane] / lane_energy_baseline[lane]
            )
            scores.append((lane, whitened_score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def _max_notes_for_time(
        self,
        rms_value: float,
        high_energy_threshold: float,
        medium_energy_threshold: float,
    ) -> int:
        if rms_value >= high_energy_threshold:
            return min(self.max_chord_size, self.high_energy_chord_size)
        if rms_value >= medium_energy_threshold:
            return min(self.max_chord_size, self.medium_energy_chord_size)
        return 1

    def _select_lanes(
        self,
        lane_scores: list[tuple[int, float]],
        max_notes_for_time: int,
    ) -> list[tuple[int, float]]:
        return lane_scores[: max(0, max_notes_for_time)]

    def _estimate_hold_end_time_ms(
        self,
        start_time_ms: int,
        next_onset_time_ms: int | None,
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
        if next_onset_time_ms is not None:
            snapped_end_time_ms = min(snapped_end_time_ms, next_onset_time_ms)
        if snapped_end_time_ms <= start_time_ms:
            return None
        if snapped_end_time_ms - start_time_ms < self.hold_min_duration_ms:
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
        "--max-chord-size",
        type=int,
        default=4,
        help="绝对高能区允许的最大按键数上限，默认 4",
    )
    parser.add_argument(
        "--high-energy-percentile",
        type=float,
        default=95.0,
        help="Top 5%% 高能区阈值分位数，默认 95",
    )
    parser.add_argument(
        "--medium-energy-percentile",
        type=float,
        default=80.0,
        help="Top 20%% 中高能区阈值分位数，默认 80",
    )
    parser.add_argument(
        "--high-energy-chord-size",
        type=int,
        default=4,
        help="绝对高能区允许的最大和弦数，默认 4",
    )
    parser.add_argument(
        "--medium-energy-chord-size",
        type=int,
        default=2,
        help="中高能区允许的最大和弦数，默认 2",
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
        max_chord_size=args.max_chord_size,
        high_energy_percentile=args.high_energy_percentile,
        medium_energy_percentile=args.medium_energy_percentile,
        high_energy_chord_size=args.high_energy_chord_size,
        medium_energy_chord_size=args.medium_energy_chord_size,
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
        f"Mapping params: key_count={args.key_count} smooth_frames={args.smooth_frames} high_energy_percentile={args.high_energy_percentile} medium_energy_percentile={args.medium_energy_percentile} high_energy_chord_size={args.high_energy_chord_size} medium_energy_chord_size={args.medium_energy_chord_size} max_chord_size={args.max_chord_size} hold_min_duration_ms={args.hold_min_duration_ms}"
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
    harmonic_samples: np.ndarray
    percussive_samples: np.ndarray
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
    y_harmonic, y_percussive = librosa.effects.hpss(y)
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
        y=_pick_onset_input(y, y_percussive),
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
            y_harmonic,
            sr=sr,
            hop_length=config.hop_length,
            bins_per_octave=config.cqt_bins_per_octave,
            n_bins=config.cqt_n_bins,
        )
    )
    chroma_cqt = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=config.hop_length,
        bins_per_octave=config.cqt_bins_per_octave,
    )

    return ExtractedFeatures(
        audio_path=audio_path,
        audio_samples=y,
        harmonic_samples=y_harmonic,
        percussive_samples=y_percussive,
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


def _pick_onset_input(y: np.ndarray, y_percussive: np.ndarray) -> np.ndarray:
    if np.any(np.abs(y_percussive) > 1e-8):
        return y_percussive
    return y
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

automakeosufile/tools/export_cyletix_song_bundle.py
```python
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

algorithm/__init__.py
```python

```

---

algorithm/边缘检测.py
```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from scipy.ndimage import binary_erosion

# 读取频谱图数据（假设是一个numpy数组）
spectrogram = np.random.rand(100, 200)

# 应用边缘检测
edges = canny(spectrogram)

# 二值腐蚀，将边缘变细
edges = binary_erosion(edges)

# 使用概率霍夫变换检测直线
lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

# 在频谱图上绘制检测到的直线
plt.imshow(spectrogram, cmap='gray')
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')

plt.title("Spectrogram with Detected Lines")
plt.show()
```

---

algorithm/binarize.py
```python
'''
Description: 二值化
Author: Cyletix
Date: 2023-03-17 21:26:19
LastEditTime: 2023-04-02 23:29:52
FilePath: \AutoMakeosuFile\binarize.py
'''
# from PIL import Image
# import pytesseract


def simple_binarize(chroma0):
    import copy
    chroma=copy.deepcopy(chroma0)#深度拷贝,不修改原变量
    threshold=0.9 #阈值
    for i in range(len(chroma)):
        for j in range(len(chroma[0])):
            if chroma[i][j]>threshold:
                chroma[i][j]=1
            else:
                chroma[i][j]=0
    return chroma

# def read_text(text_path):
#     """
#     传入文本(jpg、png)的绝对路径,读取文本
#     :param text_path:
#     :return: 文本内容
#     """
#     # 验证码图片转字符串
#     im = Image.open(text_path)
#     # 转化为8bit的黑白图片
#     imgry = im.convert('L')
#     # 二值化，采用阈值分割算法，threshold为分割点
#     threshold = 140
#     table = []
#     for j in range(256):
#         if j < threshold:
#             table.append(0)
#         else:
#             table.append(1)
#     out = imgry.point(table, '1')
#     # 识别文本
#     text = pytesseract.image_to_string(out, lang="eng", config='--psm 6')
#     return text

# %%

# 图像二值化
def threshold(self):
    import cv2 as cv
    src = self.cv_read_img(self.src_file)
    if src is None:
        return

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 这个函数的第一个参数就是原图像，原图像应该是灰度图。
    # 第二个参数就是用来对像素值进行分类的阈值。
    # 第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 第四个参数来决定阈值方法，见threshold_simple()
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    ret, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    self.decode_and_show_dst(dst)



# %%
# def local_binarize(image):
#     import cv2
#     import numpy as np

#     # 读取输入图像
#     input_image = cv2.imread(image, 0)

#     # 定义局部二值化参数
#     block_size = (3, 25)
#     constant = 2

#     # 应用局部二值化
#     output_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

#     # 显示输出图像
#     cv2.imshow('Output Image', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()








# %%
if __name__=='__main__':
    ndarray_test=[[1,2,3],[3,2,1],[1,3,2]]

    local_binarize('Back.png')


    import cv2
    # 定义局部二值化参数
    block_size = 3
    constant = 2

    #图片输入
    image='E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\Back.png'
    input_image = cv2.imread(image,0)
    output_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    cv2.imshow('Output Image', output_image)



    #向量输入
    input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 先转换为二值化期望的输入:灰度图
    gray_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
    # 应用局部二值化

    # 显示输出图像


    #映射到rgb值域
    chroma1 = (chroma*255).astype('uint8')
    threshold = cv2.adaptiveThreshold(chroma1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    plt.imshow(threshold,'gray')
```

---

algorithm/bpm_calculate.py
```python
"""
Description: bpm,first/last beat time calculate by librosa
Author: Cyletix
Date: 2023-03-16 19:15:49
LastEditTime: 2023-08-04 18:16:56
FilePath: \AutoMakeosuFile\bpm_calculate.py
"""

import librosa


def get_bpm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # print("Tempo 1:", tempo)
    first_beat_time, last_beat_time = librosa.frames_to_time(
        (beats[0], beats[-1]), sr=sr
    )
    # print("Tempo 2:", 60/((last_beat_time-first_beat_time)/(len(beats)-1)))
    tempo2 = 60 / ((last_beat_time - first_beat_time) / (len(beats) - 1))
    bpm = int(tempo2)
    print("bpm", bpm)

    return bpm, first_beat_time, last_beat_time


if __name__ == "__main__":
    filename = r"audio\NIGHTFALL.wav"
    y, sr = librosa.load(filename)
    result = get_bpm(y, sr)
    print(result)
```

---

algorithm/chatgpt_generate_function.py
```python
'''
Description: 如何用python检测音频中的波峰?
要检测音频中的波峰，首先需要对音频进行采样，然后可以使用一些数学算法来识别波峰。
一种方法是使用积分运算，即将音频信号进行平滑和求和。可以将每个采样点与它相邻的采样点相乘，然后对结果求和。如果结果是正的，则表示这个点是波峰，如果是负的，则表示这是波谷。
还有一种方法是使用高通滤波器，即使用滤波器将高频部分保留下来，并将低频部分删除。然后可以寻找信号中的极值点，作为波峰。
以下是一个简单的示例，使用积分运算识别音频中的波峰：
Author: Cyletix
Date: 2023-02-11 19:27:58
LastEditTime: 2023-02-11 19:28:02
FilePath: \AutoMakeosuFile\chatgpt生成函数.py
'''
import numpy as np


def detect_peaks(signal):
    peaks = []
    peak = False
    for i in range(1, len(signal) - 1):
        if signal[i] > 0 and signal[i-1] < 0:
            peaks.append(i)
            peak = True
        elif signal[i] < 0 and signal[i-1] > 0:
            peak = False
    return peaks

def integrate(signal):
    return np.cumsum(signal)

def detect_audio_peaks(audio_signal):
    integrated_signal = integrate(audio_signal)
    peaks = detect_peaks(integrated_signal)
    return peaks


if __name__=='__main__':
    mp3_path=''
```

---

algorithm/custom_onset_detect.py
```python
'''
Description: 计算每一个时间点的鼓点强度
Author: Cyletix
Date: 2023-03-14 17:52:37
LastEditTime: 2023-03-15 01:34:47
FilePath: \AutoMakeosuFile\onset detection function.py
'''
import librosa


def my_one_detect(filename):
    # load audio file

    y, sr = librosa.load(filename)

    # compute onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)#表示每个时间点上的音频信号中的突变程度

    # detect onsets
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    # convert onset frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)



if __name__=='__main__':
    filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'
```

---

algorithm/custom_stft.py
```python
'''
Description: STFT变换,已尝试成功
Author: Cyletix
Date: 2023-02-12 00:23:32
LastEditTime: 2023-03-15 04:52:01
FilePath: \AutoMakeosuFile\custom_stft.py
'''

import librosa
import numpy as np


def my_stft(filename):
    # Load audio signal
    y, sr = librosa.load(filename)

    # Compute the spectrogram
    n_fft = 2048
    hop_length = 512
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    frequencies = np.linspace(0, sr / 2, spectrogram.shape[0])
    times = librosa.times_like(spectrogram[0,:])

    return y,sr,n_fft,hop_length,spectrogram,frequencies,times


def my_plot(times,frequencies,spectrogram):
    import matplotlib.pyplot as plt

    # Plot the spectrogram as a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(times, frequencies)
    ax.plot_surface(X, Y, spectrogram)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Magnitude')
    plt.show()





# # 将STFT转换为分贝
# stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# # 绘制频谱图
# plt.figure(figsize=(12, 6))
# librosa.display.specshow(stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('STFT Magnitude')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()



if __name__=='__main__':
    filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'
    
    y,sr,n_fft,hop_length,spectrogram,frequencies,times = my_stft(filename)
    
    my_plot(times,frequencies,spectrogram)
```

---

algorithm/dynamic_spectrum.py
```python
'''
Description: 这个不知道怎么换成自己的音频,长度不知道怎么确定,
Author: Cyletix
Date: 2022-06-01 19:07:54
LastEditTime: 2023-03-17 04:40:14
FilePath: \AutoMakeosuFile\dynamic_spectrum.py
'''
#!/usr/bin/env python
# 
# (c) 2017 Juha Vierinen
import matplotlib.pyplot as plt
import numpy as n
import scipy.signal as s

plt.style.use('dark_background')#设置plot风格


# create dynamic spectrum
def spectrogram(x,M=1024,N=128,delta_n=100):
    max_t=int(n.floor((len(x)-N)/delta_n))
    t=n.arange(max_t)
    X=n.zeros([max_t,M],dtype=n.complex64)
    w=s.hann(N)
    xin=n.zeros(N)
    for i in range(max_t):
        xin[0:N]=x[i*delta_n+n.arange(N)]
        X[i,:]=n.fft.fft(w*xin,M)
    return(X)

# sample rate (Hz)
fs=4096.0

# sample indexes (one second of signal)
nn=n.arange(4096)
# generate a chirp signal
x=n.sin(0.15e-14*nn**5.0)

# time step
delta_n=25
M=2048
# create dynamic spectrum.
# Use
# - 2048 point FFT
# - 128 samples for each spectra
# - 100 sample increments in time
S=spectrogram(x,M=M,N=128,delta_n=delta_n)
freqs=n.fft.fftfreq(2048,d=1.0/fs)
time=delta_n*n.arange(S.shape[0])/fs



# plot signal
plt.figure(figsize=(12,10))
plt.subplot(211)
plt.plot(nn/fs,x)
plt.title("Signal $x[n]$")
plt.xlabel("Time (s)")
plt.ylabel("Signal amplitude")

plt.subplot(212)
plt.title("Spectrogram")
plt.pcolormesh(time,freqs[0:(M//2)],n.transpose(10.0*n.log10(n.abs(S[:,0:(M//2)])**2.0)),vmin=0)
plt.xlim([0,n.max(time)])
plt.ylim([0,fs/2.0])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
cb=plt.colorbar(orientation="horizontal")
cb.set_label("dB")
plt.tight_layout()
plt.savefig("dynspec.png")
plt.show()
```

---

algorithm/HandleData.py
```python
"""
复刻自Java项目Free
[Android Studio 开发实践——简易版音游APP（一）_androidstudio gameactivity-CSDN博客]
(https://blog.csdn.net/qq_43533416/article/details/105631991)
[[E:\GitHub\Free]]
"""

import numpy as np
from scipy.fft import fft


def handle_data(
    data,
    sampling_rate,
    music_time,
    window_size=1024,
    threshold_window_size=20,
    multiplier=3.0,
):
    """
    识别节奏点的简化Python版本。

    参数:
    - data: 输入的音频数据。
    - sampling_rate: 采样率。似乎这里没有用到
    - music_time: 音乐总时间，单位为毫秒。
    - window_size: 分析窗口的大小。
    - threshold_window_size: 计算阈值时考虑的周围窗口数量。
    - multiplier: 阈值乘数。
    """
    # 初始化变量
    spectral_flux = []  # 光谱通量
    threshold = []  # 阈值
    all_time = []  # 节奏点时间

    # 按窗口遍历数据
    for i in range(0, len(data) - window_size, window_size):
        # 对当前窗口进行FFT
        window_data = data[i : i + window_size]
        fft_result = np.abs(fft(window_data))

        # 计算光谱通量
        if i == 0:
            flux = sum(fft_result)
        else:
            flux = sum(np.abs(fft_result - prev_fft_result))
        spectral_flux.append(flux)

        prev_fft_result = fft_result

    # 计算阈值和检测节奏点
    for i in range(len(spectral_flux)):
        start = max(0, i - threshold_window_size)
        end = min(len(spectral_flux) - 1, i + threshold_window_size)
        local_mean = np.mean(spectral_flux[start : end + 1])
        threshold.append(local_mean * multiplier)

        if spectral_flux[i] > threshold[i]:
            time = int(i * window_size / (len(data) * 1.0) * music_time)
            if len(all_time) == 0 or (time - all_time[-1]) > 100:  # 100ms防抖动
                all_time.append(time)

    return all_time
```

---

algorithm/nmf_separator.py
```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF


@dataclass(slots=True)
class ComponentStats:
    index: int
    activation_mean: float
    activation_std: float
    activation_flux: float
    smoothness_score: float
    spectral_centroid_bin: float
    removal_score: float


@dataclass(slots=True)
class NMFSeparationResult:
    audio_path: Path
    sample_rate: int
    n_fft: int
    hop_length: int
    n_components: int
    removed_components: list[int]
    kept_components: list[int]
    residual_wav_path: Path
    removed_wav_path: Path
    activation_matrix_path: Path
    basis_matrix_path: Path
    summary_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 STFT + NMF + Masking + iSTFT 进行音色分离，并导出残差音频"
    )
    parser.add_argument("--audio", type=Path, required=True, help="输入音频路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "nmf",
        help="输出目录，默认 output/nmf",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="重采样采样率，默认 22050",
    )
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument(
        "--n-components",
        type=int,
        default=8,
        help="NMF 分解分量数，默认 8",
    )
    parser.add_argument(
        "--remove-count",
        type=int,
        default=2,
        help="按启发式自动移除的持续性分量数，默认 2",
    )
    parser.add_argument(
        "--remove-components",
        default="",
        help="显式指定要移除的分量索引，例如 0,3,5；传空则自动选择",
    )
    parser.add_argument(
        "--mask-power",
        type=float,
        default=2.0,
        help="软掩码幂次，默认 2.0",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="NMF 最大迭代数，默认 400",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="NMF 随机种子，默认 0",
    )
    return parser


def separate_audio_with_nmf(
    audio_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_components: int = 8,
    remove_count: int = 2,
    remove_components: list[int] | None = None,
    mask_power: float = 2.0,
    max_iter: int = 400,
    random_state: int = 0,
) -> NMFSeparationResult:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    sr = int(sr)
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_complex).astype(np.float64)
    phase = np.exp(1j * np.angle(stft_complex))

    basis, activation = _factorize_magnitude(
        magnitude=magnitude,
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
    )
    component_stack = _build_component_stack(basis=basis, activation=activation)
    component_stats = _analyze_components(component_stack, activation)

    chosen_components = (
        sorted(set(remove_components))
        if remove_components
        else _select_components_to_remove(component_stats, remove_count)
    )
    kept_components = [
        index for index in range(n_components) if index not in chosen_components
    ]

    removed_mask, residual_mask = _build_masks(
        component_stack=component_stack,
        removed_components=chosen_components,
        mask_power=mask_power,
    )

    removed_audio = _reconstruct_audio(removed_mask, magnitude, phase, hop_length)
    residual_audio = _reconstruct_audio(residual_mask, magnitude, phase, hop_length)

    stem = _safe_stem(audio_path)
    residual_wav_path = output_dir / f"{stem}_residual.wav"
    removed_wav_path = output_dir / f"{stem}_removed.wav"
    activation_matrix_path = output_dir / f"{stem}_activation_matrix.npy"
    basis_matrix_path = output_dir / f"{stem}_basis_matrix.npy"
    summary_path = output_dir / f"{stem}_nmf_summary.json"

    sf.write(residual_wav_path, residual_audio, sr)
    sf.write(removed_wav_path, removed_audio, sr)
    np.save(activation_matrix_path, activation)
    np.save(basis_matrix_path, basis)

    summary_payload = {
        "audio_path": str(audio_path),
        "sample_rate": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_components": n_components,
        "remove_count": remove_count,
        "removed_components": chosen_components,
        "kept_components": kept_components,
        "mask_power": mask_power,
        "component_stats": [asdict(item) for item in component_stats],
        "outputs": {
            "residual_wav": str(residual_wav_path),
            "removed_wav": str(removed_wav_path),
            "activation_matrix": str(activation_matrix_path),
            "basis_matrix": str(basis_matrix_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return NMFSeparationResult(
        audio_path=audio_path,
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_components=n_components,
        removed_components=chosen_components,
        kept_components=kept_components,
        residual_wav_path=residual_wav_path,
        removed_wav_path=removed_wav_path,
        activation_matrix_path=activation_matrix_path,
        basis_matrix_path=basis_matrix_path,
        summary_path=summary_path,
    )


def _factorize_magnitude(
    magnitude: np.ndarray,
    n_components: int,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    safe_magnitude = np.maximum(magnitude, 1e-10)
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        random_state=random_state,
    )
    basis = model.fit_transform(safe_magnitude)
    activation = model.components_
    return basis, activation


def _build_component_stack(basis: np.ndarray, activation: np.ndarray) -> np.ndarray:
    n_components = activation.shape[0]
    component_stack = np.zeros(
        (n_components, basis.shape[0], activation.shape[1]), dtype=np.float64
    )
    for index in range(n_components):
        component_stack[index] = np.outer(basis[:, index], activation[index])
    return component_stack


def _analyze_components(
    component_stack: np.ndarray,
    activation: np.ndarray,
) -> list[ComponentStats]:
    stats: list[ComponentStats] = []
    for index in range(component_stack.shape[0]):
        activation_curve = activation[index]
        normalized = activation_curve / (np.mean(activation_curve) + 1e-8)
        flux = (
            float(np.mean(np.abs(np.diff(normalized)))) if len(normalized) > 1 else 0.0
        )
        smoothness = float(1.0 / (flux + 1e-6))
        spectral_profile = np.mean(component_stack[index], axis=1)
        bins = np.arange(len(spectral_profile), dtype=np.float64)
        centroid = float(
            np.sum(bins * spectral_profile) / (np.sum(spectral_profile) + 1e-8)
        )
        removal_score = float(smoothness * (1.0 + np.mean(normalized)))
        stats.append(
            ComponentStats(
                index=index,
                activation_mean=float(np.mean(activation_curve)),
                activation_std=float(np.std(activation_curve)),
                activation_flux=flux,
                smoothness_score=smoothness,
                spectral_centroid_bin=centroid,
                removal_score=removal_score,
            )
        )
    stats.sort(key=lambda item: item.removal_score, reverse=True)
    return stats


def _select_components_to_remove(
    component_stats: list[ComponentStats], remove_count: int
) -> list[int]:
    remove_count = max(0, min(remove_count, len(component_stats)))
    return sorted(item.index for item in component_stats[:remove_count])


def _build_masks(
    component_stack: np.ndarray,
    removed_components: list[int],
    mask_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.sum(component_stack, axis=0) + 1e-10
    removed = (
        np.sum(component_stack[removed_components], axis=0)
        if removed_components
        else np.zeros_like(total)
    )
    kept = np.maximum(total - removed, 0.0)

    removed_mask = np.power(removed, mask_power) / (
        np.power(removed, mask_power) + np.power(kept, mask_power) + 1e-10
    )
    residual_mask = 1.0 - removed_mask
    return removed_mask, residual_mask


def _reconstruct_audio(
    soft_mask: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_length: int,
) -> np.ndarray:
    masked_complex = soft_mask * magnitude * phase
    waveform = librosa.istft(masked_complex, hop_length=hop_length)
    return np.asarray(waveform, dtype=np.float32)


def _parse_component_indices(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(int(text))
    return values


def _safe_stem(path: Path) -> str:
    name = path.stem
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name


def main() -> int:
    args = build_parser().parse_args()
    remove_components = _parse_component_indices(args.remove_components)
    result = separate_audio_with_nmf(
        audio_path=args.audio,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_components=args.n_components,
        remove_count=args.remove_count,
        remove_components=remove_components,
        mask_power=args.mask_power,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    print(f"Residual wav: {result.residual_wav_path}")
    print(f"Removed wav: {result.removed_wav_path}")
    print(f"Activation matrix: {result.activation_matrix_path}")
    print(f"Basis matrix: {result.basis_matrix_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Removed components: {result.removed_components}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

algorithm/note_duration.py
```python
"""
计算指定bpm,fb,lb的时间划分网格,并提供画图功能
did not finished
"""

import numpy as np

# %% 测试向量推算16分音大小
time_point = [
    690,
    1749,
    2455,
    3161,
    3513,
    3690,
    4043,
    4219,
    4572,
    4925,
    5102,
    5455,
    5631,
    5984,
    6160,
    6337,
    6425,
    6513,
    6602,
    6866,
    7043,
    7219,
    7396,
    7749,
    7925,
    8102,
    8278,
    8455,
    8631,
    8808,
    8984,
    9160,
    9249,
    9337,
    9425,
    9690,
    9866,
    10043,
    10219,
]
time_point_diff = np.diff(time_point)
cent16_interval = min(np.unique(time_point_diff))  # 88ms  16分音


def note_duration(bpm, first_beat_time, last_beat_time):
    # %% 判断点面类型
    interval = bpm / 60 * 1000

    cent4_interval = bpm / 60 * 1000 / 4
    cent8_interval = bpm / 60 * 1000 / 8
    cent16_interval = bpm / 60 * 1000 / 16
    cent32_interval = bpm / 60 * 1000 / 16

    cent4 = np.arange(first_beat_time, last_beat_time, 88 * 4 / 1000)
    cent4 = np.arange(first_beat_time, last_beat_time, 88 * 4 / 1000)
    cent16 = np.arange(first_beat_time, last_beat_time, 88 / 1000)
    return interval


def plot_note_duration(bpm, first_beat_time, last_beat_time, ax):
    import matplotlib.pyplot as plt

    color_group = ["red", "orange", "yellow", "green", "blue", "purple"]
    for i in range(6):
        cent_interval = interval / 2 ** (i + 1)
        print(str(2 ** (i)) + "分音:", cent_interval)
        cent = np.arange(first_beat_time, last_beat_time, cent_interval)
        ax[3].vlines(
            cent, ymin=0, ymax=7 - i, linestyles="dashed", colors=color_group[i]
        )  # 竖线
    ax[3].vlines(cent16, ymin=0, ymax=1, linestyles="dashed", colors="blue")  # 竖线
    # 画图
    plt.show()
```

---

algorithm/PCA_test.py
```python
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# %% 生成信号
def rand_spectrogram():
    # 生成一个虚拟的频谱图，假设有12个频率，每个频率在100个时间步上的强度
    n_freq = 12
    n_time = 100
    spectrogram = np.random.rand(n_freq, n_time)

    # 初始化PCA对象，指定降维后的维度为6
    pca = PCA(n_components=6)


    # 在频谱图上拟合PCA模型，需要将频率和时间维度进行转置
    pca.fit(spectrogram.T)



# 读取音频文件
filename = 'dragon_girl.mp3'
y, sr = librosa.load(filename)


# %% 计算处理
# 计算Chroma频谱图
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# 初始化PCA对象，指定降维后的维度
pca = PCA(n_components=10)

# 在Chroma频谱图上拟合PCA模型
pca.fit(chroma.T)

# 对Chroma频谱图进行降维
reduced_chroma = pca.transform(chroma.T)



# %% 画图
# 绘制原始Chroma频谱图和降维后的Chroma频谱图，同步x轴
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.imshow(chroma, aspect='auto', cmap='viridis', origin='lower')
ax1.set_title("Original Chroma Spectrogram")

ax2.imshow(reduced_chroma.T, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Reduced Chroma Spectrogram")

plt.tight_layout()
plt.show()
```

---

algorithm/svd.py
```python
'''
Description: 文件描述
Author: Cyletix
Date: 2023-04-02 23:38:20
LastEditTime: 2023-04-02 23:41:58
FilePath: \AutoMakeosuFile\svd.py
'''
import numpy as np

def svd_decomp(m, n, m1, n1):
    # Create a random matrix of size m x n
    A = np.random.rand(m, n)
    # Perform SVD decomposition
    U, s, V = np.linalg.svd(A)
    # Construct a diagonal matrix with singular values
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)
    # Construct the output matrix
    B = U[:, :m1] @ S[:m1, :n1] @ V[:n1, :]
    return B[:m1,:n1]



# Example usage
if __name__=='__main__':
    B = svd_decomp(5, 4, 3, 2)
    print(B)
```

---

algorithm/windows_size.py
```python
def calculate_windows_size(bpm, note_division, sample_rate=44100):
    # 计算时间间隔（秒）
    time_interval = 60 / (bpm * note_division)
    # 确定窗口大小
    window_size = int(time_interval * sample_rate)
    return window_size


if __name__ == "__main__":
    import librosa

    # 示例参数
    bpm = 120  # BPM值
    sample_rate = 44100  # 采样率
    note_divisions = [2, 4, 8, 16]  # 分音值

    # 计算不同分音的窗口大小
    window_sizes = [
        calculate_windows_size(bpm, nd, sample_rate) for nd in note_divisions
    ]

    # 打印窗口大小
    for nd, ws in zip(note_divisions, window_sizes):
        print(f"{nd}分音的窗口大小: {ws} 样本点（约{ws/sample_rate}秒）")

    # 对音频信号进行STFT（示例）
    audio_file = "path/to/your/audio/file.wav"
    y, sr = librosa.load(audio_file, sr=sample_rate)
    for ws in window_sizes:
        stft_result = librosa.stft(y, n_fft=ws)
        # 处理STFT结果...
```

---

alignment/__init__.py
```python
from .offset_aligner import align_osu_to_audio, estimate_best_offset, main

__all__ = ["align_osu_to_audio", "estimate_best_offset", "main"]
```

---

alignment/offset_aligner.py
```python
from automakeosufile.alignment.offset_aligner import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
```

---

training/__init__.py
```python
__all__: list[str] = []
```

---

training/build_frame_labels.py
```python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from automakeosufile.config import FeatureConfig, dense_onset_config
from automakeosufile.features.extractor import ExtractedFeatures, extract_features
from automakeosufile.parsers.osu_mania import OsuFileData, parse_osu_file
from automakeosufile.tools.inspect_sample import _apply_onset_overrides


@dataclass(slots=True)
class FrameDatasetResult:
    output_path: Path
    metadata_path: Path
    x_shape: tuple[int, int]
    y_shape: tuple[int, int]
    note_count: int
    feature_names: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把特征与对齐后的 GT 谱面打包成帧级训练数据集"
    )
    parser.add_argument(
        "--alignment-json", type=Path, required=True, help="offset_alignment.json 路径"
    )
    parser.add_argument("--osu", type=Path, help="可选：显式指定参考 osu 路径")
    parser.add_argument("--audio", type=Path, help="可选：显式指定音频路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "training",
        help="输出目录，默认 output/training",
    )
    parser.add_argument(
        "--dense-onset", action="store_true", help="使用高密度 onset 配置提特征"
    )
    parser.add_argument("--onset-delta", type=float)
    parser.add_argument("--onset-wait", type=int)
    parser.add_argument("--onset-pre-max", type=int)
    parser.add_argument("--onset-post-max", type=int)
    parser.add_argument("--onset-pre-avg", type=int)
    parser.add_argument("--onset-post-avg", type=int)
    parser.add_argument("--onset-aggregate", choices=["mean", "median"])
    return parser


def build_frame_dataset(
    alignment_json_path: str | Path,
    output_dir: str | Path,
    osu_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    config: FeatureConfig | None = None,
) -> FrameDatasetResult:
    alignment_json_path = Path(alignment_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_payload = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    osu_path = Path(osu_path or alignment_payload["osu_path"])
    audio_path = Path(audio_path or alignment_payload["audio_path"])

    osu_data = parse_osu_file(osu_path)
    features = extract_features(audio_path, config=config)
    aligned_times_ms = alignment_payload["alignment"]["aligned_times_ms"]

    if len(aligned_times_ms) != len(osu_data.hit_objects):
        raise ValueError(
            f"aligned_times_ms 数量({len(aligned_times_ms)})与 hit_objects 数量({len(osu_data.hit_objects)})不一致"
        )

    x_matrix, feature_names = _build_feature_matrix(features)
    y_matrix = _build_label_matrix(
        osu_data=osu_data,
        aligned_times_ms=aligned_times_ms,
        frame_count=x_matrix.shape[0],
        sample_rate=features.sample_rate,
        hop_length=(config.hop_length if config else FeatureConfig().hop_length),
    )

    stem = _safe_stem(osu_path)
    dataset_path = output_dir / stem / "dataset_frames.npz"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_path.with_name("dataset_frames_meta.json")

    np.savez_compressed(
        dataset_path,
        X=x_matrix.astype(np.float32),
        Y=y_matrix.astype(np.uint8),
        aligned_times_ms=np.asarray(aligned_times_ms, dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype="<U64"),
    )

    metadata = {
        "alignment_json": str(alignment_json_path),
        "osu_path": str(osu_path),
        "audio_path": str(audio_path),
        "x_shape": list(x_matrix.shape),
        "y_shape": list(y_matrix.shape),
        "note_count": len(osu_data.hit_objects),
        "key_count": osu_data.key_count,
        "feature_names": feature_names,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return FrameDatasetResult(
        output_path=dataset_path,
        metadata_path=metadata_path,
        x_shape=tuple(x_matrix.shape),
        y_shape=tuple(y_matrix.shape),
        note_count=len(osu_data.hit_objects),
        feature_names=feature_names,
    )


def _build_feature_matrix(features: ExtractedFeatures) -> tuple[np.ndarray, list[str]]:
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    cqt = features.cqt_magnitude.T
    mel = features.mel_db.T
    chroma = features.chroma_cqt.T
    rms = features.rms.reshape(-1, 1)
    onset = features.onset_envelope.reshape(-1, 1)

    frame_count = min(len(cqt), len(mel), len(chroma), len(rms), len(onset))
    cqt = cqt[:frame_count]
    mel = mel[:frame_count]
    chroma = chroma[:frame_count]
    rms = rms[:frame_count]
    onset = onset[:frame_count]

    feature_blocks.extend([cqt, mel, chroma, rms, onset])
    feature_names.extend(
        [
            *[f"cqt_{index}" for index in range(cqt.shape[1])],
            *[f"mel_db_{index}" for index in range(mel.shape[1])],
            *[f"chroma_{index}" for index in range(chroma.shape[1])],
            "rms",
            "onset_envelope",
        ]
    )

    x_matrix = np.concatenate(feature_blocks, axis=1)
    return x_matrix, feature_names


def _build_label_matrix(
    osu_data: OsuFileData,
    aligned_times_ms: list[float],
    frame_count: int,
    sample_rate: int,
    hop_length: int,
) -> np.ndarray:
    y_matrix = np.zeros((frame_count, osu_data.key_count), dtype=np.uint8)
    for hit_object, aligned_time_ms in zip(osu_data.hit_objects, aligned_times_ms):
        frame_index = int(round((float(aligned_time_ms) / 1000.0) * sample_rate / hop_length))
        frame_index = min(frame_count - 1, max(0, frame_index))
        y_matrix[frame_index, hit_object.lane] = 1
    return y_matrix


def _safe_stem(path: Path) -> str:
    name = path.stem
    for char in '<>:"/\\|?*':
        name = name.replace(char, '_')
    return name


def main() -> int:
    args = build_parser().parse_args()
    config = dense_onset_config() if args.dense_onset else FeatureConfig()
    config = _apply_onset_overrides(config, args)
    result = build_frame_dataset(
        alignment_json_path=args.alignment_json,
        output_dir=args.output_dir,
        osu_path=args.osu,
        audio_path=args.audio,
        config=config,
    )
    print(f"Dataset: {result.output_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"X shape: {result.x_shape}")
    print(f"Y shape: {result.y_shape}")
    print(f"Note count: {result.note_count}")
    print(f"Feature dims: {len(result.feature_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

