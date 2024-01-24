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