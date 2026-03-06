'''
Description: 这是librosa官网的一些使用例子,不重要
Author: Cyletix
Date: 2023-03-15 01:33:46
LastEditTime: 2023-03-15 04:51:39
FilePath: \AutoMakeosuFile\librosa_example.py
'''
import librosa
import matplotlib.pyplot as plt
import numpy as np

#首先，加载一些音频并绘制频谱图

filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'
y, sr = librosa.load(filename, duration=3)
D = np.abs(librosa.stft(y))
times = librosa.times_like(D)
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()


#构建标准起始函数
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
ax[1].plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
           label='Mean (mel)')

# 中值聚合和自定义 mel 选项
onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.median,
                                         fmax=8000, n_mels=256)
ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
           label='Median (custom mel)')

# 恒定 Q 谱图而不是 Mel
C = np.abs(librosa.cqt(y=y, sr=sr))
onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
         label='Mean (CQT)')
ax[1].legend()
ax[1].set(ylabel='Normalized strength', yticks=[])


plt.show()
          
input('按任意键继续')