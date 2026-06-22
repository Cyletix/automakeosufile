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
