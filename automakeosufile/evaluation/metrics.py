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
