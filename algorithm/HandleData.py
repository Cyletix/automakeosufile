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
