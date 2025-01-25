"""
Description: 从音频文件生成频谱图
Author: Cyletix
Date: 2022-02-01 01:50:24
LastEditTime: 2023-02-11 19:30:33
FilePath: \AutoMakeosuFile\readmp3.py
"""

import os
import numpy as np
from numpy.fft import rfft, ifft, fftfreq
import librosa
import matplotlib.pyplot as plt


def analyze_audio(audio_path):
    # 加载音频文件
    y, sr = librosa.load(audio_path)

    # 生成时间序列
    t = np.arange(0, len(y)) / sr

    # 生成离散傅里叶变换采样频率
    w = fftfreq(len(t), d=1 / sr)

    # 傅里叶变换，得到所有频率的幅值
    frames_fft = rfft(y)

    # 绘制原始频谱，红色
    plt.plot(w, frames_fft, color="r")

    # 等比例减小所有幅值
    frames_fft = frames_fft * 0.3

    # 使用正弦曲线调整音乐音量
    frames_fft = np.array(tuple(map(lambda x, y: x * y, frames_fft, np.sin(t))))

    # 绘制减小幅值之后的频谱，蓝色
    plt.plot(w, frames_fft, color="b")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 使用audio目录下的第一个wav文件
    audio_dir = os.path.join(os.path.dirname(__file__), "..", "audio")
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    if wav_files:
        audio_path = os.path.join(audio_dir, wav_files[0])
        analyze_audio(audio_path)
    else:
        print("No WAV files found in audio directory")
