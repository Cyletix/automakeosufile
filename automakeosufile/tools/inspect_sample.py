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
