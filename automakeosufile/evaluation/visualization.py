from __future__ import annotations

from pathlib import Path
from math import ceil

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from automakeosufile.config import FeatureConfig
from automakeosufile.features.extractor import ExtractedFeatures
from automakeosufile.parsers.osu_mania import OsuFileData


def save_onset_overlay_plot(
    output_path: str | Path,
    features: ExtractedFeatures,
    osu_data: OsuFileData,
    config: FeatureConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_objects = _limit_objects(osu_data, config)
    gt_times_sec = np.array([obj.time_ms / 1000 for obj in gt_objects])
    onset_times_sec = features.onset_times
    envelope_times = (
        np.arange(len(features.onset_envelope))
        * config.hop_length
        / features.sample_rate
    )
    lane_times = np.array([obj.time_ms / 1000 for obj in gt_objects])
    lane_values = np.array([obj.lane for obj in gt_objects])
    hold_mask = np.array([obj.is_hold for obj in gt_objects])

    window_seconds = config.visualize_window_seconds
    segment_count = max(1, ceil(features.duration_seconds / window_seconds))
    fig, axes = plt.subplots(
        nrows=segment_count * 2,
        ncols=1,
        figsize=(18, max(6, segment_count * 3.5)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)

    envelope_peak = (
        float(np.max(features.onset_envelope)) if len(features.onset_envelope) else 1.0
    )

    for segment_index in range(segment_count):
        start_sec = segment_index * window_seconds
        end_sec = min(features.duration_seconds, start_sec + window_seconds)
        envelope_ax = axes[segment_index * 2]
        lane_ax = axes[segment_index * 2 + 1]

        envelope_mask = (envelope_times >= start_sec) & (envelope_times < end_sec)
        gt_mask = (gt_times_sec >= start_sec) & (gt_times_sec < end_sec)
        onset_mask = (onset_times_sec >= start_sec) & (onset_times_sec < end_sec)
        lane_mask = (lane_times >= start_sec) & (lane_times < end_sec)

        envelope_ax.plot(
            envelope_times[envelope_mask],
            features.onset_envelope[envelope_mask],
            color="cyan",
            linewidth=1.0,
        )
        envelope_ax.vlines(
            gt_times_sec[gt_mask],
            ymin=0,
            ymax=envelope_peak,
            colors="orange",
            alpha=0.3,
            linewidth=0.8,
        )
        envelope_ax.vlines(
            onset_times_sec[onset_mask],
            ymin=0,
            ymax=envelope_peak,
            colors="lime",
            alpha=0.35,
            linewidth=0.8,
        )
        envelope_ax.set_xlim(start_sec, end_sec)
        envelope_ax.set_ylabel("Onset")
        envelope_ax.set_title(f"Onset overlay  {start_sec:.0f}s - {end_sec:.0f}s")

        lane_ax.scatter(
            lane_times[lane_mask],
            lane_values[lane_mask],
            s=12,
            c=np.where(hold_mask[lane_mask], "tomato", "gold"),
            alpha=0.9,
            edgecolors="none",
        )
        lane_ax.set_xlim(start_sec, end_sec)
        lane_ax.set_ylabel("Lane")
        lane_ax.set_xlabel("Time (s)")
        lane_ax.set_ylim(-0.5, max(0.5, osu_data.key_count - 0.5))
        lane_ax.set_title("GT note raster (tap=gold, hold=red)")
        lane_ax.set_facecolor("#111111")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_chroma_overlay_plot(
    output_path: str | Path,
    features: ExtractedFeatures,
    osu_data: OsuFileData,
    config: FeatureConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_objects = _limit_objects(osu_data, config)
    gt_times_sec = np.array([obj.time_ms / 1000 for obj in gt_objects])
    lane_values = np.array([obj.lane for obj in gt_objects])
    chroma_times = (
        np.arange(features.chroma_cqt.shape[1])
        * config.hop_length
        / features.sample_rate
    )
    window_seconds = config.visualize_window_seconds
    segment_count = max(1, ceil(features.duration_seconds / window_seconds))

    fig, axes = plt.subplots(
        nrows=segment_count * 2,
        ncols=1,
        figsize=(18, max(7, segment_count * 3.8)),
        sharex=False,
        height_ratios=[3.2, 1.0] * segment_count,
    )
    axes = np.atleast_1d(axes)

    for segment_index in range(segment_count):
        start_sec = segment_index * window_seconds
        end_sec = min(features.duration_seconds, start_sec + window_seconds)
        chroma_ax = axes[segment_index * 2]
        lane_ax = axes[segment_index * 2 + 1]

        frame_mask = (chroma_times >= start_sec) & (chroma_times < end_sec)
        gt_mask = (gt_times_sec >= start_sec) & (gt_times_sec < end_sec)

        if np.any(frame_mask):
            chroma_slice = features.chroma_cqt[:, frame_mask]
            chroma_ax.imshow(
                chroma_slice,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[start_sec, end_sec, 0, 12],
            )
        chroma_ax.vlines(
            gt_times_sec[gt_mask],
            ymin=0,
            ymax=11.5,
            colors="white",
            alpha=0.18,
            linewidth=0.7,
        )
        chroma_ax.set_xlim(start_sec, end_sec)
        chroma_ax.set_ylim(0, 12)
        chroma_ax.set_ylabel("Chroma")
        chroma_ax.set_title(f"Chroma overlay  {start_sec:.0f}s - {end_sec:.0f}s")

        lane_ax.scatter(
            gt_times_sec[gt_mask],
            lane_values[gt_mask],
            s=12,
            c="gold",
            alpha=0.9,
            edgecolors="none",
        )
        lane_ax.set_xlim(start_sec, end_sec)
        lane_ax.set_ylim(-0.5, max(0.5, osu_data.key_count - 0.5))
        lane_ax.set_ylabel("Lane")
        lane_ax.set_xlabel("Time (s)")
        lane_ax.set_title("GT note lanes")
        lane_ax.set_facecolor("#111111")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _limit_objects(osu_data: OsuFileData, config: FeatureConfig):
    if config.max_visualize_notes is None:
        return osu_data.hit_objects
    return osu_data.hit_objects[: config.max_visualize_notes]
