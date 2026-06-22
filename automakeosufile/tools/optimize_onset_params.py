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
