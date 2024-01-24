'''
Description: 文件描述
Author: Cyletix
Date: 2023-02-11 19:31:14
LastEditTime: 2023-03-17 21:23:17
FilePath: \AutoMakeosuFile\main.py
'''

import librosa
import matplotlib.pyplot as plt



# %%

plt.style.use('dark_background')#设置plot风格

filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'

# Load a wav file
y, sr = librosa.load(filename)

# Compute mel power spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to log scale (dB)
log_S = librosa.power_to_db(S)


# %% 画图
# Plot the spectrogram
# plt.figure()
# librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel")
# plt.colorbar(format="%+2.0f dB")
# plt.title("Mel power spectrogram")
# plt.tight_layout()
# plt.savefig("test.png")
# plt.show()

# %%

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
librosa.display.specshow(log_S, sr=sr, x_axis="Time [s]", y_axis="Freq [Hz]",ax=ax[0])
ax.set(xlim=[0, 10], ylim=[0, 100])

# compute onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# detect onsets
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

# convert onset frames to times
onset_times = librosa.frames_to_time(onset_frames, sr=sr)


# ax.vlines(onset_times, 0, 1, linestyles='dashed', colors='red')


fig.show()



# %%

# #与上面功能几乎一致
# onset_frames = librosa.onset.onset_detect(y=y)
# D = librosa.stft(y)
# librosa.display.specshow(librosa.amplitude_to_db(D))
# plt.vlines(onset_frames, 0, sr, color='r', linestyle='--')
# plt.show()

pitch = librosa.piptrack(y=y, sr=sr)


# %% chroma_cqt
chroma=librosa.feature.chroma_cqt(y=y,sr=sr)

librosa.display.specshow(chroma,x_axis='time',y_axis='chroma')
plt.colorbar()
