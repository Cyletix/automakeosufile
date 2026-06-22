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
