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
