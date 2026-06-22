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
