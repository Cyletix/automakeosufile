"""
Description: STFT变换,已尝试成功
Author: Cyletix
Date: 2023-02-12 00:23:32
LastEditTime: 2023-03-15 04:52:01
FilePath: \AutoMakeosuFile\custom_stft.py
"""

import librosa
import numpy as np


def my_stft(filename):
    # Load audio signal
    y, sr = librosa.load(filename)

    # Compute the spectrogram
    n_fft = 2048
    hop_length = int(sr * 0.05859)  # 32分音符间隔 (58.59ms)
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    frequencies = np.linspace(0, sr / 2, spectrogram.shape[0])
    times = librosa.times_like(spectrogram[0, :])

    return y, sr, n_fft, hop_length, spectrogram, frequencies, times


def my_plot(times, frequencies, spectrogram, output_path=None):
    import matplotlib.pyplot as plt

    # Convert to dB scale
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    # Create 2D plot
    plt.figure(figsize=(12, 6))
    plt.imshow(
        spectrogram_db,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
        cmap="viridis",
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate spectrogram from audio file")
    parser.add_argument(
        "--input",
        type=str,
        default="audio/dragon_girl.wav",
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/spectrogram.png",
        help="Path to save output image",
    )
    args = parser.parse_args()

    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        y, sr, n_fft, hop_length, spectrogram, frequencies, times = my_stft(args.input)
        my_plot(times, frequencies, spectrogram, args.output)
        print(f"Spectrogram saved to {args.output}")
    except Exception as e:
        print(f"Error processing audio file: {e}")
