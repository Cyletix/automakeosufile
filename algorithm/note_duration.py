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
