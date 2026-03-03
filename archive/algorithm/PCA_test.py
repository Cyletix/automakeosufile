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