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
        next_onset_time_by_time = self._build_next_onset_time_lookup(snapped_times_ms)
        harmonic_cqt = features.cqt_magnitude
        lane_energy = self._lane_energy_matrix(harmonic_cqt)
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
        remaining_capacity = self.max_chord_size - len(occupied)
        if remaining_capacity <= 0:
            return []
        selected: list[int] = []
        top_score = lane_scores[0][1] if lane_scores else 0.0

        for lane, score in lane_scores:
            if lane not in occupied:
                selected.append(lane)
                top_score = score
                break

        if not selected:
            return []

        if len(selected) >= remaining_capacity:
            return selected

        if len(lane_scores) > 1 and top_score > 0:
            for lane, score in lane_scores:
                if lane in occupied or lane == selected[0]:
                    continue
                if len(selected) >= remaining_capacity:
                    break
                if (
                    score >= top_score * self.chord_relative_threshold
                    or score >= absolute_lane_threshold
                ):
                    selected.append(lane)
                if len(selected) >= remaining_capacity:
                    break

        return selected

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
