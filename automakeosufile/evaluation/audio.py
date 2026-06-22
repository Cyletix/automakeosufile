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
