### custom_stft.py
```
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

```

### custom_onset_detect.py
```
'''
Description: 计算每一个时间点的鼓点强度
Author: Cyletix
Date: 2023-03-14 17:52:37
LastEditTime: 2023-03-15 01:34:47
FilePath: \AutoMakeosuFile\onset detection function.py
'''
import librosa


def my_one_detect(filename):
    # load audio file

    y, sr = librosa.load(filename)

    # compute onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)#表示每个时间点上的音频信号中的突变程度

    # detect onsets
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    # convert onset frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)



if __name__=='__main__':
    filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'
```

### bpm_calculate.py
```
import librosa


def get_bpm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # print("Tempo 1:", tempo)
    first_beat_time, last_beat_time = librosa.frames_to_time(
        (beats[0], beats[-1]), sr=sr
    )
    # print("Tempo 2:", 60/((last_beat_time-first_beat_time)/(len(beats)-1)))
    tempo2 = 60 / ((last_beat_time - first_beat_time) / (len(beats) - 1))
    bpm = int(tempo2)
    print("bpm", bpm)

    return bpm, first_beat_time, last_beat_time


if __name__ == "__main__":
    filename = r"audio\NIGHTFALL.wav"
    y, sr = librosa.load(filename)
    result = get_bpm(y, sr)
    print(result)

```

### note_duration.py
```
"""
计算指定bpm,fb,lb的时间划分网格,并提供画图功能
did not finished
"""

import numpy as np

# %% 测试向量推算16分音大小
time_point = [
    690,
    1749,
    2455,
    3161,
    3513,
    3690,
    4043,
    4219,
    4572,
    4925,
    5102,
    5455,
    5631,
    5984,
    6160,
    6337,
    6425,
    6513,
    6602,
    6866,
    7043,
    7219,
    7396,
    7749,
    7925,
    8102,
    8278,
    8455,
    8631,
    8808,
    8984,
    9160,
    9249,
    9337,
    9425,
    9690,
    9866,
    10043,
    10219,
]
time_point_diff = np.diff(time_point)
cent16_interval = min(np.unique(time_point_diff))  # 88ms  16分音


def note_duration(bpm, first_beat_time, last_beat_time):
    # %% 判断点面类型
    interval = bpm / 60 * 1000

    cent4_interval = bpm / 60 * 1000 / 4
    cent8_interval = bpm / 60 * 1000 / 8
    cent16_interval = bpm / 60 * 1000 / 16
    cent32_interval = bpm / 60 * 1000 / 16

    cent4 = np.arange(first_beat_time, last_beat_time, 88 * 4 / 1000)
    cent4 = np.arange(first_beat_time, last_beat_time, 88 * 4 / 1000)
    cent16 = np.arange(first_beat_time, last_beat_time, 88 / 1000)
    return interval


def plot_note_duration(bpm, first_beat_time, last_beat_time, ax):
    import matplotlib.pyplot as plt

    color_group = ["red", "orange", "yellow", "green", "blue", "purple"]
    for i in range(6):
        cent_interval = interval / 2 ** (i + 1)
        print(str(2 ** (i)) + "分音:", cent_interval)
        cent = np.arange(first_beat_time, last_beat_time, cent_interval)
        ax[3].vlines(
            cent, ymin=0, ymax=7 - i, linestyles="dashed", colors=color_group[i]
        )  # 竖线
    ax[3].vlines(cent16, ymin=0, ymax=1, linestyles="dashed", colors="blue")  # 竖线
    # 画图
    plt.show()

```

### windows_size.py
```
def calculate_windows_size(bpm, note_division, sample_rate=44100):
    # 计算时间间隔（秒）
    time_interval = 60 / (bpm * note_division)
    # 确定窗口大小
    window_size = int(time_interval * sample_rate)
    return window_size


if __name__ == "__main__":
    import librosa

    # 示例参数
    bpm = 120  # BPM值
    sample_rate = 44100  # 采样率
    note_divisions = [2, 4, 8, 16]  # 分音值

    # 计算不同分音的窗口大小
    window_sizes = [
        calculate_windows_size(bpm, nd, sample_rate) for nd in note_divisions
    ]

    # 打印窗口大小
    for nd, ws in zip(note_divisions, window_sizes):
        print(f"{nd}分音的窗口大小: {ws} 样本点（约{ws/sample_rate}秒）")

    # 对音频信号进行STFT（示例）
    audio_file = "path/to/your/audio/file.wav"
    y, sr = librosa.load(audio_file, sr=sample_rate)
    for ws in window_sizes:
        stft_result = librosa.stft(y, n_fft=ws)
        # 处理STFT结果...

```

### __init__.py
```

```

### dynamic_spectrum.py
```
'''
Description: 这个不知道怎么换成自己的音频,长度不知道怎么确定,
Author: Cyletix
Date: 2022-06-01 19:07:54
LastEditTime: 2023-03-17 04:40:14
FilePath: \AutoMakeosuFile\dynamic_spectrum.py
'''
#!/usr/bin/env python
# 
# (c) 2017 Juha Vierinen
import matplotlib.pyplot as plt
import numpy as n
import scipy.signal as s

plt.style.use('dark_background')#设置plot风格


# create dynamic spectrum
def spectrogram(x,M=1024,N=128,delta_n=100):
    max_t=int(n.floor((len(x)-N)/delta_n))
    t=n.arange(max_t)
    X=n.zeros([max_t,M],dtype=n.complex64)
    w=s.hann(N)
    xin=n.zeros(N)
    for i in range(max_t):
        xin[0:N]=x[i*delta_n+n.arange(N)]
        X[i,:]=n.fft.fft(w*xin,M)
    return(X)

# sample rate (Hz)
fs=4096.0

# sample indexes (one second of signal)
nn=n.arange(4096)
# generate a chirp signal
x=n.sin(0.15e-14*nn**5.0)

# time step
delta_n=25
M=2048
# create dynamic spectrum.
# Use
# - 2048 point FFT
# - 128 samples for each spectra
# - 100 sample increments in time
S=spectrogram(x,M=M,N=128,delta_n=delta_n)
freqs=n.fft.fftfreq(2048,d=1.0/fs)
time=delta_n*n.arange(S.shape[0])/fs



# plot signal
plt.figure(figsize=(12,10))
plt.subplot(211)
plt.plot(nn/fs,x)
plt.title("Signal $x[n]$")
plt.xlabel("Time (s)")
plt.ylabel("Signal amplitude")

plt.subplot(212)
plt.title("Spectrogram")
plt.pcolormesh(time,freqs[0:(M//2)],n.transpose(10.0*n.log10(n.abs(S[:,0:(M//2)])**2.0)),vmin=0)
plt.xlim([0,n.max(time)])
plt.ylim([0,fs/2.0])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
cb=plt.colorbar(orientation="horizontal")
cb.set_label("dB")
plt.tight_layout()
plt.savefig("dynspec.png")
plt.show()
```

### main.py
```
import os
import numpy as np
import librosa
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
n_fft = 1024  # samples per frame (reduced for better time resolution)
hop_length = 256  # samples between frames (reduced for better time resolution)
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# 将STFT转换为分贝
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
# 绘制频谱图


# %%fig绘图
# 创建2D绘图
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
librosa.display.specshow(
    stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear", ax=ax[0]
)
ax[0].set_ylim([0, 500])
ax[0].set_title("stft_db")

# %%plt绘图
# plt.figure(figsize=(12, 6))
# librosa.display.specshow(
#     stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear"
# )
# plt.colorbar(format="%+2.0f dB")
# plt.title("STFT Magnitude")
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()


# %% stft 3D
# n_fft = 2048
# hop_length = 512
# spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
# frequencies = np.linspace(0, sr / 2, spectrogram.shape[0])
# times = librosa.times_like(spectrogram[0, :])

# # 创建单独的3D图
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(111, projection="3d")
# X, Y = np.meshgrid(times, frequencies)
# ax3d.plot_surface(X, Y, spectrogram)
# ax3d.set_xlabel("Time (s)")
# ax3d.set_ylabel("Frequency (Hz)")
# ax3d.set_zlabel("Magnitude")
# plt.show()


# %% mel
# Compute mel power spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to log scale (dB)
log_S = librosa.power_to_db(S)

librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel", fmax=8000, ax=ax[1])

onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# detect onsets with improved parameters
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    units="time",
    pre_max=0.03,  # 30ms
    post_max=0.03,  # 30ms
    pre_avg=0.1,  # 100ms
    post_avg=0.1,  # 100ms
    delta=0.07,  # threshold
    wait=0.03,  # 30ms
)

# convert onset frames to times
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
ax[1].set_title("mel power spectrogram")


# %% chroma_cqt
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
librosa.display.specshow(
    chroma, x_axis="time", y_axis="chroma", ax=ax[2], x_coords=times
)
ax[2].set_title("chroma_cqt (aligned)")


# %%设置x轴范围
x_max_set = np.max(len(chroma)) / 2
plt.xlim(0, x_max_set)


# %% 二值化的chroma
chroma_b = simple_binarize(chroma)
librosa.display.specshow(
    chroma_b, x_axis="time", y_axis="chroma", ax=ax[3], x_coords=times
)
ax[3].set_title("chroma_cqt binarize (aligned)")
ax[3].set_ylim([0, 24])  # Set y-axis limits to match chroma resolution
ax[3].set_ylabel("Pitch Class")

# Print first 10 seconds of chroma_b data
print("\nChroma_b data structure (first 10 seconds):")
time_slice = int(10 * sr / hop_length)  # Calculate number of frames for 10 seconds
print(chroma_b[:, :time_slice])


# Show all plots after all are configured
plt.tight_layout()
plt.show()

# %% bpm计算
# 使用更准确的bpm检测方法
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
bpm = int(tempo)


print(f"Using BPM: {bpm}")

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
beatmap.set_difficulty(hp=5, cs=4, od=5, ar=5)

# Generate hit objects from onsets with improved parameters
beatmap.generate_from_analysis(
    bpm=bpm,
    hit_times=onset_times * 1000,  # Convert to milliseconds
    duration=librosa.get_duration(y=y, sr=sr),
    density=0.8,  # Adjust note density
    pattern_variation=0.3,  # Add some variation to patterns
    column_count=6,  # Set to 4 columns for standard mania
)

# Save beatmap
output_file = os.path.splitext(filename)[0] + ".osu"
beatmap.save(output_file)
print(f"Saved beatmap to {output_file}")

```

### chatgpt_generate_function.py
```
'''
Description: 如何用python检测音频中的波峰?
要检测音频中的波峰，首先需要对音频进行采样，然后可以使用一些数学算法来识别波峰。
一种方法是使用积分运算，即将音频信号进行平滑和求和。可以将每个采样点与它相邻的采样点相乘，然后对结果求和。如果结果是正的，则表示这个点是波峰，如果是负的，则表示这是波谷。
还有一种方法是使用高通滤波器，即使用滤波器将高频部分保留下来，并将低频部分删除。然后可以寻找信号中的极值点，作为波峰。
以下是一个简单的示例，使用积分运算识别音频中的波峰：
Author: Cyletix
Date: 2023-02-11 19:27:58
LastEditTime: 2023-02-11 19:28:02
FilePath: \AutoMakeosuFile\chatgpt生成函数.py
'''
import numpy as np


def detect_peaks(signal):
    peaks = []
    peak = False
    for i in range(1, len(signal) - 1):
        if signal[i] > 0 and signal[i-1] < 0:
            peaks.append(i)
            peak = True
        elif signal[i] < 0 and signal[i-1] > 0:
            peak = False
    return peaks

def integrate(signal):
    return np.cumsum(signal)

def detect_audio_peaks(audio_signal):
    integrated_signal = integrate(audio_signal)
    peaks = detect_peaks(integrated_signal)
    return peaks


if __name__=='__main__':
    mp3_path=''
```

### binarize.py
```
'''
Description: 二值化
Author: Cyletix
Date: 2023-03-17 21:26:19
LastEditTime: 2023-04-02 23:29:52
FilePath: \AutoMakeosuFile\binarize.py
'''
# from PIL import Image
# import pytesseract


def simple_binarize(chroma0):
    import copy
    chroma=copy.deepcopy(chroma0)#深度拷贝,不修改原变量
    threshold=0.9 #阈值
    for i in range(len(chroma)):
        for j in range(len(chroma[0])):
            if chroma[i][j]>threshold:
                chroma[i][j]=1
            else:
                chroma[i][j]=0
    return chroma

# def read_text(text_path):
#     """
#     传入文本(jpg、png)的绝对路径,读取文本
#     :param text_path:
#     :return: 文本内容
#     """
#     # 验证码图片转字符串
#     im = Image.open(text_path)
#     # 转化为8bit的黑白图片
#     imgry = im.convert('L')
#     # 二值化，采用阈值分割算法，threshold为分割点
#     threshold = 140
#     table = []
#     for j in range(256):
#         if j < threshold:
#             table.append(0)
#         else:
#             table.append(1)
#     out = imgry.point(table, '1')
#     # 识别文本
#     text = pytesseract.image_to_string(out, lang="eng", config='--psm 6')
#     return text

# %%

# 图像二值化
def threshold(self):
    import cv2 as cv
    src = self.cv_read_img(self.src_file)
    if src is None:
        return

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 这个函数的第一个参数就是原图像，原图像应该是灰度图。
    # 第二个参数就是用来对像素值进行分类的阈值。
    # 第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 第四个参数来决定阈值方法，见threshold_simple()
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    ret, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    self.decode_and_show_dst(dst)



# %%
# def local_binarize(image):
#     import cv2
#     import numpy as np

#     # 读取输入图像
#     input_image = cv2.imread(image, 0)

#     # 定义局部二值化参数
#     block_size = (3, 25)
#     constant = 2

#     # 应用局部二值化
#     output_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

#     # 显示输出图像
#     cv2.imshow('Output Image', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()








# %%
if __name__=='__main__':
    ndarray_test=[[1,2,3],[3,2,1],[1,3,2]]

    local_binarize('Back.png')


    import cv2
    # 定义局部二值化参数
    block_size = 3
    constant = 2

    #图片输入
    image='E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\Back.png'
    input_image = cv2.imread(image,0)
    output_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    cv2.imshow('Output Image', output_image)



    #向量输入
    input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 先转换为二值化期望的输入:灰度图
    gray_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2GRAY)
    # 应用局部二值化

    # 显示输出图像


    #映射到rgb值域
    chroma1 = (chroma*255).astype('uint8')
    threshold = cv2.adaptiveThreshold(chroma1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    plt.imshow(threshold,'gray')

```

### 边缘检测.py
```
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from scipy.ndimage import binary_erosion

# 读取频谱图数据（假设是一个numpy数组）
spectrogram = np.random.rand(100, 200)

# 应用边缘检测
edges = canny(spectrogram)

# 二值腐蚀，将边缘变细
edges = binary_erosion(edges)

# 使用概率霍夫变换检测直线
lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

# 在频谱图上绘制检测到的直线
plt.imshow(spectrogram, cmap='gray')
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')

plt.title("Spectrogram with Detected Lines")
plt.show()

```

### mp3_to_wav.py
```
import os
import librosa
import soundfile


def mp32wav(mp3_file):
    fname, bac = os.path.splitext(mp3_file)
    if bac == ".mp3":
        y, sr = librosa.load(mp3_file)
        wav_file = os.path.splitext(mp3_file)[0] + ".wav"
        soundfile.write(wav_file, y, sr)
        return wav_file
    else:
        print("this file is not mp3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    # filename = r'E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\YELL!.mp3'

    # 转换audio文件夹下所有文件
    audio_directory = os.path.join(os.path.dirname(__file__), "..", "audio")
    for filename in os.listdir(audio_directory):
        file_path = os.path.join(audio_directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".mp3"):
            wav_file = os.path.splitext(file_path)[0] + ".wav"
            if not os.path.exists(wav_file):
                mp32wav(file_path)

```

### PCA_test.py
```
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# %% 生成信号
def rand_spectrogram():
    # 生成一个虚拟的频谱图，假设有12个频率，每个频率在100个时间步上的强度
    n_freq = 12
    n_time = 100
    spectrogram = np.random.rand(n_freq, n_time)

    # 初始化PCA对象，指定降维后的维度为6
    pca = PCA(n_components=6)


    # 在频谱图上拟合PCA模型，需要将频率和时间维度进行转置
    pca.fit(spectrogram.T)



# 读取音频文件
filename = 'dragon_girl.mp3'
y, sr = librosa.load(filename)


# %% 计算处理
# 计算Chroma频谱图
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# 初始化PCA对象，指定降维后的维度
pca = PCA(n_components=10)

# 在Chroma频谱图上拟合PCA模型
pca.fit(chroma.T)

# 对Chroma频谱图进行降维
reduced_chroma = pca.transform(chroma.T)



# %% 画图
# 绘制原始Chroma频谱图和降维后的Chroma频谱图，同步x轴
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.imshow(chroma, aspect='auto', cmap='viridis', origin='lower')
ax1.set_title("Original Chroma Spectrogram")

ax2.imshow(reduced_chroma.T, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Reduced Chroma Spectrogram")

plt.tight_layout()
plt.show()
```

### HandleData.py
```
"""
复刻自Java项目Free
[Android Studio 开发实践——简易版音游APP（一）_androidstudio gameactivity-CSDN博客]
(https://blog.csdn.net/qq_43533416/article/details/105631991)
[[E:\GitHub\Free]]
"""

import numpy as np
from scipy.fft import fft


def handle_data(
    data,
    sampling_rate,
    music_time,
    window_size=1024,
    threshold_window_size=20,
    multiplier=3.0,
):
    """
    识别节奏点的简化Python版本。

    参数:
    - data: 输入的音频数据。
    - sampling_rate: 采样率。似乎这里没有用到
    - music_time: 音乐总时间，单位为毫秒。
    - window_size: 分析窗口的大小。
    - threshold_window_size: 计算阈值时考虑的周围窗口数量。
    - multiplier: 阈值乘数。
    """
    # 初始化变量
    spectral_flux = []  # 光谱通量
    threshold = []  # 阈值
    all_time = []  # 节奏点时间

    # 按窗口遍历数据
    for i in range(0, len(data) - window_size, window_size):
        # 对当前窗口进行FFT
        window_data = data[i : i + window_size]
        fft_result = np.abs(fft(window_data))

        # 计算光谱通量
        if i == 0:
            flux = sum(fft_result)
        else:
            flux = sum(np.abs(fft_result - prev_fft_result))
        spectral_flux.append(flux)

        prev_fft_result = fft_result

    # 计算阈值和检测节奏点
    for i in range(len(spectral_flux)):
        start = max(0, i - threshold_window_size)
        end = min(len(spectral_flux) - 1, i + threshold_window_size)
        local_mean = np.mean(spectral_flux[start : end + 1])
        threshold.append(local_mean * multiplier)

        if spectral_flux[i] > threshold[i]:
            time = int(i * window_size / (len(data) * 1.0) * music_time)
            if len(all_time) == 0 or (time - all_time[-1]) > 100:  # 100ms防抖动
                all_time.append(time)

    return all_time

```

### svd.py
```
'''
Description: 文件描述
Author: Cyletix
Date: 2023-04-02 23:38:20
LastEditTime: 2023-04-02 23:41:58
FilePath: \AutoMakeosuFile\svd.py
'''
import numpy as np

def svd_decomp(m, n, m1, n1):
    # Create a random matrix of size m x n
    A = np.random.rand(m, n)
    # Perform SVD decomposition
    U, s, V = np.linalg.svd(A)
    # Construct a diagonal matrix with singular values
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)
    # Construct the output matrix
    B = U[:, :m1] @ S[:m1, :n1] @ V[:n1, :]
    return B[:m1,:n1]



# Example usage
if __name__=='__main__':
    B = svd_decomp(5, 4, 3, 2)
    print(B)
```

### osu_file_make.py
```
import os


class OSUGenerator:
    def __init__(self):
        self.metadata = {
            "Title": "",
            "Artist": "",
            "Creator": "",
            "Version": "",
            "AudioFilename": "",
            "PreviewTime": -1,
            "BeatmapID": 0,
            "BeatmapSetID": -1,
        }
        self.difficulty = {
            "HPDrainRate": 5,
            "CircleSize": 4,
            "OverallDifficulty": 5,
            "ApproachRate": 5,
            "SliderMultiplier": 1.4,
            "SliderTickRate": 1,
        }
        self.timing_points = []
        self.hit_objects = []

    def set_audio_file(self, filename):
        self.metadata["AudioFilename"] = os.path.basename(filename)

    def set_metadata(self, title, artist, creator, version):
        """Set metadata for the beatmap."""
        self.metadata.update(
            {
                "Title": title,
                "TitleUnicode": title,
                "Artist": artist,
                "ArtistUnicode": artist,
                "Creator": creator,
                "Version": version,
                "Source": "",
                "Tags": "",
                "BeatmapID": 0,
                "BeatmapSetID": -1,
            }
        )

    def set_difficulty(self, hp, cs, od, ar):
        self.difficulty["HPDrainRate"] = hp
        self.difficulty["CircleSize"] = cs
        self.difficulty["OverallDifficulty"] = od
        self.difficulty["ApproachRate"] = ar

    def generate_from_analysis(self, bpm, hit_times, duration):
        # Add timing point
        self.timing_points.append([0, 60000 / bpm, 4, 2, 1, 60, 1, 0])

        # Add hit objects
        for time in hit_times:
            # Spread notes across 4 columns for 4K mode
            x = 64 + (int(time / 100) % 4) * 128
            y = 192  # Fixed y position for mania
            self.hit_objects.append([x, y, int(time), 1, 0, "0:0:0:0:"])

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            # Write metadata
            f.write("osu file format v14\n\n")
            f.write("[General]\n")
            f.write(f"AudioFilename: {self.metadata['AudioFilename']}\n")
            f.write(f"PreviewTime: {self.metadata['PreviewTime']}\n")
            f.write("Mode: 3\n")
            f.write("Countdown: 1\n")
            f.write("SampleSet: Soft\n")
            f.write("StackLeniency: 0.7\n")
            f.write("LetterboxInBreaks: 0\n")
            f.write("SpecialStyle: 0\n")
            f.write("WidescreenStoryboard: 0\n\n")

            # Write metadata section
            f.write("[Metadata]\n")
            for key, value in self.metadata.items():
                if key != "AudioFilename" and key != "PreviewTime":
                    f.write(f"{key}:{value}\n")
            f.write("\n")

            # Write difficulty settings
            f.write("[Difficulty]\n")
            for key, value in self.difficulty.items():
                f.write(f"{key}:{value}\n")
            f.write("\n")

            # Write timing points
            f.write("[TimingPoints]\n")
            for point in self.timing_points:
                f.write(",".join(map(str, point)) + "\n")
            f.write("\n")

            # Write hit objects
            f.write("[HitObjects]\n")
            for obj in self.hit_objects:
                f.write(",".join(map(str, obj)) + "\n")


# Example usage
if __name__ == "__main__":
    generator = OSUGenerator()
    generator.set_audio_file("example.mp3")
    generator.set_metadata(
        "Example Title", "Example Artist", "Example Creator", "Example Version"
    )
    generator.set_difficulty(5, 4, 5, 5)
    generator.generate_from_analysis(120, [1000, 2000, 3000], 4000)
    generator.save("example.osu")

```

