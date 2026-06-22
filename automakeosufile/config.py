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
