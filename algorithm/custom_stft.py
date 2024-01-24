'''
Description: STFT变换,已尝试成功
Author: Cyletix
Date: 2023-02-12 00:23:32
LastEditTime: 2023-03-15 04:52:01
FilePath: \AutoMakeosuFile\custom_stft.py
'''

import librosa
import numpy as np


def my_stft(filename):
    # Load audio signal
    y, sr = librosa.load(filename)

    # Compute the spectrogram
    n_fft = 2048
    hop_length = 512
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    frequencies = np.linspace(0, sr / 2, spectrogram.shape[0])
    times = librosa.times_like(spectrogram[0,:])

    return y,sr,n_fft,hop_length,spectrogram,frequencies,times


def my_plot(times,frequencies,spectrogram):
    import matplotlib.pyplot as plt

    # Plot the spectrogram as a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(times, frequencies)
    ax.plot_surface(X, Y, spectrogram)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Magnitude')
    plt.show()





# # 将STFT转换为分贝
# stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# # 绘制频谱图
# plt.figure(figsize=(12, 6))
# librosa.display.specshow(stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('STFT Magnitude')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()



if __name__=='__main__':
    filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'
    
    y,sr,n_fft,hop_length,spectrogram,frequencies,times = my_stft(filename)
    
    my_plot(times,frequencies,spectrogram)