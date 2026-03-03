print("Starting script...")
import os
import numpy as np
import librosa
from sklearn.decomposition import PCA

from algorithm.binarize import simple_binarize
from algorithm.bpm_calculate import get_bpm
from algorithm.windows_size import calculate_windows_size
from algorithm.custom_onset_detect import my_one_detect
from fileprocess.mp3_to_wav import mp32wav
from fileprocess.osu_file_make import OSUGenerator

# import cv2
# from IPython.display import Audio
# import seaborn
# import scipy
# import mir_eval


# %% load music
print("Loading music...")
# filename = 'E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\YELL!.wav'
# filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\VELVET CLOAK - Ryunosuke Kudo An - Immobility.mp3'
# filename = 'dragon_girl.wav'
filename = "audio/Epilogue.mp3"
wav_filename = os.path.splitext(filename)[0] + ".wav"
if not os.path.exists(wav_filename):
    print("Converting mp3 to wav...")
    mp32wav(filename, wav_filename)
else:
    filename = wav_filename

y, sr = librosa.load(filename, duration=30.0)
print("Music loaded.")

# Audio(data=y,rate=sr)


# 波形图
# librosa.display.waveshow(y)


# %%stft
print("Calculating STFT...")
n_fft = 1024  # samples per frame (reduced for better time resolution)
hop_length = 256  # samples between frames (reduced for better time resolution)
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# 将STFT转换为分贝
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
print("STFT calculated.")
# 绘制频谱图


# %% mel
print("Calculating Mel spectrogram...")
# Compute mel power spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to log scale (dB)
log_S = librosa.power_to_db(S)
print("Mel spectrogram calculated.")

print("Detecting onsets...")
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# detect onsets with improved parameters
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    units="frames",
    pre_max=0.03,  # 30ms
    post_max=0.03,  # 30ms
    pre_avg=0.1,  # 100ms
    post_avg=0.1,  # 100ms
    delta=0.07,  # threshold
    wait=0.03,  # 30ms
)

# convert onset frames to times
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
print("Onsets detected.")


# %% chroma_cqt
print("Calculating Chroma CQT...")
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
print("Chroma CQT calculated.")


# %% 二值化的chroma
print("Binarizing Chroma...")
chroma_b = simple_binarize(chroma)
print("Chroma binarized.")

# Print first 10 seconds of chroma_b data
print("\nChroma_b data structure (first 10 seconds):")
time_slice = int(10 * sr / hop_length)  # Calculate number of frames for 10 seconds
print(chroma_b[:, :time_slice])


# %% bpm计算
print("Calculating BPM...")
# 使用更准确的bpm检测方法
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
bpm = int(tempo)
print(f"Using BPM: {bpm}")


print("Generating beatmap...")
# %% Generate osu! beatmap
from fileprocess.osu_file_make import OSUGenerator
from fileprocess.osu_file_parse import OSUData

# Create beatmap
beatmap = OSUGenerator()
beatmap.set_audio_file(filename)
# Set metadata using MP3 filename
filename_base = os.path.splitext(os.path.basename(filename))[0]
beatmap.set_metadata(
    title=filename_base,
    artist=filename_base,
    creator="AutoMakeosuFile",
    version="Auto v1.0",
)
beatmap.set_difficulty(hp=5, cs=7, od=5, ar=5)

# Generate hit objects from onsets with improved parameters
beatmap.generate_from_analysis(
    bpm=bpm,
    hit_times=onset_times * 1000,  # Convert to milliseconds
    duration=librosa.get_duration(y=y, sr=sr),
    chroma_data=chroma_b,
    sr=sr,
    hop_length=hop_length,
    density=0.8,  # Adjust note density
    pattern_variation=0.3,  # Add some variation to patterns
    column_count=7,  # Set to 7 columns for standard mania
)

# Save beatmap
output_file = os.path.splitext(filename)[0] + ".osu"
beatmap.save(output_file)
print(f"Saved beatmap to {output_file}")
print("Script finished.")
