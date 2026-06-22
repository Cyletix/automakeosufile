from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from automakeosufile.config import FeatureConfig


@dataclass(slots=True)
class ExtractedFeatures:
    audio_path: Path
    audio_samples: np.ndarray
    harmonic_samples: np.ndarray
    percussive_samples: np.ndarray
    duration_seconds: float
    sample_rate: int
    stft_complex: np.ndarray
    stft_magnitude: np.ndarray
    stft_db: np.ndarray
    mel_power: np.ndarray
    mel_db: np.ndarray
    rms: np.ndarray
    onset_envelope: np.ndarray
    onset_frames: np.ndarray
    onset_times: np.ndarray
    cqt_magnitude: np.ndarray
    chroma_cqt: np.ndarray


def extract_features(
    audio_path: str | Path, config: FeatureConfig | None = None
) -> ExtractedFeatures:
    config = config or FeatureConfig()
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=config.sample_rate)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    aggregate = _get_onset_aggregate(config.onset_aggregate)

    stft_complex = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    stft_magnitude = np.abs(stft_complex)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmax=config.fmax,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    rms = librosa.feature.rms(
        S=stft_magnitude, frame_length=config.n_fft, hop_length=config.hop_length
    )[0]

    onset_envelope = librosa.onset.onset_strength(
        y=_pick_onset_input(y, y_percussive),
        sr=sr,
        hop_length=config.hop_length,
        aggregate=aggregate,
    )
    onset_detect_kwargs = {
        "onset_envelope": onset_envelope,
        "sr": sr,
        "hop_length": config.hop_length,
        "backtrack": config.onset_backtrack,
    }
    if config.onset_pre_max is not None:
        onset_detect_kwargs["pre_max"] = config.onset_pre_max
    if config.onset_post_max is not None:
        onset_detect_kwargs["post_max"] = config.onset_post_max
    if config.onset_pre_avg is not None:
        onset_detect_kwargs["pre_avg"] = config.onset_pre_avg
    if config.onset_post_avg is not None:
        onset_detect_kwargs["post_avg"] = config.onset_post_avg
    if config.onset_wait is not None:
        onset_detect_kwargs["wait"] = config.onset_wait
    if config.onset_delta is not None:
        onset_detect_kwargs["delta"] = config.onset_delta

    onset_frames = librosa.onset.onset_detect(**onset_detect_kwargs)
    onset_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=config.hop_length
    )

    cqt_magnitude = np.abs(
        librosa.cqt(
            y_harmonic,
            sr=sr,
            hop_length=config.hop_length,
            bins_per_octave=config.cqt_bins_per_octave,
            n_bins=config.cqt_n_bins,
        )
    )
    chroma_cqt = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=config.hop_length,
        bins_per_octave=config.cqt_bins_per_octave,
    )

    return ExtractedFeatures(
        audio_path=audio_path,
        audio_samples=y,
        harmonic_samples=y_harmonic,
        percussive_samples=y_percussive,
        duration_seconds=float(librosa.get_duration(y=y, sr=sr)),
        sample_rate=int(sr),
        stft_complex=stft_complex,
        stft_magnitude=stft_magnitude,
        stft_db=stft_db,
        mel_power=mel_power,
        mel_db=mel_db,
        rms=rms,
        onset_envelope=onset_envelope,
        onset_frames=onset_frames,
        onset_times=onset_times,
        cqt_magnitude=cqt_magnitude,
        chroma_cqt=chroma_cqt,
    )


def _get_onset_aggregate(name: str):
    normalized = name.strip().lower()
    if normalized == "mean":
        return np.mean
    return np.median


def _pick_onset_input(y: np.ndarray, y_percussive: np.ndarray) -> np.ndarray:
    if np.any(np.abs(y_percussive) > 1e-8):
        return y_percussive
    return y
