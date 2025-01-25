"""
Description: 文件描述
Author: Cyletix
Date: 2023-03-17 21:24:23
LastEditTime: 2023-08-04 23:16:12
FilePath: \AutoMakeosuFile\main.py
"""

import os
import numpy as np
import librosa
import pywt
import matplotlib.pyplot as plt

plt.style.use("dark_background")  # 设置plot风格
from sklearn.decomposition import PCA

from algorithm.binarize import simple_binarize
from algorithm.bpm_calculate import get_bpm
from algorithm.windows_size import calculate_windows_size
from algorithm.custom_onset_detect import my_one_detect
from fileprocess.mp3_to_wav import mp32wav
from fileprocess.osu_file_make import OSUGenerator
from plotfunction.stft_plotly import plotly_plot

# import cv2
# from IPython.display import Audio
# import seaborn
# import scipy
# import mir_eval


# %% load music
# filename = 'E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\YELL!.wav'
# filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\VELVET CLOAK - Ryunosuke Kudo An - Immobility.mp3'
# filename = 'dragon_girl.wav'
filename = "audio/NIGHTFALL.mp3"

if not os.path.exists(filename):
    mp32wav(filename)

y, sr = librosa.load(filename)

# Audio(data=y,rate=sr)


# 波形图
# librosa.display.waveshow(y)


# %%stft
n_fft = 2048  # samples per frame
hop_length = 512  # samples between frames
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# 将STFT转换为分贝
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
# 绘制频谱图


# %%fig绘图
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
# fig.canvas.manager.window.showMaximized()#窗口最大化
librosa.display.specshow(
    stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear", ax=ax[0]
)
# Zpos = np.ma.masked_less(stft_db, 0)
# Zneg = np.ma.masked_greater(stft_db, 0)
# pos = ax[0].imshow(Zpos, cmap='Blues', interpolation='none')
# neg = ax[0].imshow(Zneg, cmap='Blues', interpolation='none')
# fig.colorbar(neg,format='%+2.0f dB', ax=ax[0])
# ax[0].set_xlim([0, 10])
ax[0].set_ylim([0, 500])
ax[0].set_title("stft_db")

# %%plt绘图
plt.figure(figsize=(12, 6))
librosa.display.specshow(
    stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear"
)
plt.colorbar(format="%+2.0f dB")
plt.title("STFT Magnitude")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# %% stft 3D
n_fft = 2048
hop_length = 512
spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
frequencies = np.linspace(0, sr / 2, spectrogram.shape[0])
times = librosa.times_like(spectrogram[0, :])
# # 3D图
fig3d = plt.figure()
ax = fig3d.add_subplot(313, projection="3d")
X, Y = np.meshgrid(times, frequencies)
ax.plot_surface(X, Y, spectrogram)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_zlabel("Magnitude")
plt.show()


# %% mel
# Compute mel power spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to log scale (dB)
log_S = librosa.power_to_db(S)

# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel", fmax=8000, ax=ax[1])

onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# detect onsets
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

# convert onset frames to times
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
# fig.show()
ax[1].vlines(onset_times, ymin=0, ymax=4096, linestyles="dashed", colors="red")  # 竖线
ax[1].set_title("mel power spectrogram")


# %% chroma_cqt
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
# fig, ax = plt.subplots(nrows=3, ncols=1,sharex=True)
librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", ax=ax[2])
ax[2].set_title("chroma_cqt")
# cbar = fig.colorbar(ax[2].imshow(chroma), ax=ax[2])
# cbar = fig.colorbar(ax[3].imshow(chroma_b), ax=ax[3])


# %%设置x轴范围
x_max_set = np.max(len(chroma)) / 2
plt.xlim(0, x_max_set)


# %% 二值化的chroma
chroma_b = simple_binarize(chroma)
librosa.display.specshow(chroma_b, x_axis="time", y_axis="chroma", ax=ax[3])
ax[3].set_title("chroma_cqt binarize")


# %% bpm计算
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
# print("Tempo 1:", tempo)
first_beat_time, last_beat_time = librosa.frames_to_time((beats[0], beats[-1]), sr=sr)
# print("Tempo 2:", 60/((last_beat_time-first_beat_time)/(len(beats)-1)))
tempo2 = 60 / ((last_beat_time - first_beat_time) / (len(beats) - 1))
bpm = int(tempo2)

# %% Generate osu! beatmap
from fileprocess.osu_file_make import OSUGenerator
from fileprocess.osu_file_parse import OSUData

# Create beatmap
beatmap = OSUGenerator()
beatmap.set_audio_file(filename)
beatmap.set_metadata(
    title="Auto Generated",
    artist="Unknown",
    creator="AutoMakeosuFile",
    version="Auto v1.0",
)
beatmap.set_difficulty(hp=5, cs=4, od=5, ar=5)

# Generate hit objects from onsets
beatmap.generate_from_analysis(
    bpm=bpm,
    hit_times=onset_times * 1000,  # Convert to milliseconds
    duration=librosa.get_duration(y=y, sr=sr),
)

# Save beatmap
output_file = os.path.splitext(filename)[0] + ".osu"
beatmap.save(output_file)
print(f"Saved beatmap to {output_file}")
