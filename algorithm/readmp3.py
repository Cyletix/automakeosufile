'''
Description: 文件描述
Author: Cyletix
Date: 2022-02-01 01:50:24
LastEditTime: 2023-02-11 19:30:33
FilePath: \AutoMakeosuFile\readmp3.py
'''
from operator import mul
import numpy as np
from numpy.fft import rfft, ifft, fftfreq
from moviepy.editor import Audiofileclip
from moviepy.audio.AudioClip import AudioArrayClip
import matplotlib. pyplot as plt
fn=r'C:/ Python38/一首歌.mp3'
audio =Audiofileclip(fn)
t=np.arange(0, audio. duration, 1/audio. fps)
#生成离散傅里叶变换采样频率
w= fft.freq(t size, d=t[1]-t[0])
#把音频数据转换为数组
frames = audio.to_soundarray()
# 傅里叶变换，得到所有频率的幅值
frames_fft=rfft(frames)
#绘制原始频谱，红色
plt.plot(w, frames_fft, color=r)
# 等比例减小所有幅值
frames_fft= frames_fft* 0.3
#使用正弦曲线调整音乐音量
frames_fft =np.array(tuple(map(mul, frames_fft, np. sin(t))))
#绘制减小幅值之后的频谱，蓝色
plt.plot (w, frames_fft, color=b)
#傅里叶反变换，生成并保存音频文件
frames_ifft =ifft(frames_fft).real

Audioarrayclip(frames_ifft#只用使用 frames*0.3可以实现同样效果
fps=audio. fps).write audiofile(music mp3
#显示图形
plt.show()
