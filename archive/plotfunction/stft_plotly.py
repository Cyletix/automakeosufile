import plotly.graph_objs as go

# 使用plotly绘制热力图
def plotly_plot(stft_db):
    fig = go.Figure(
        data=go.Heatmap(z=stft_db, x=time_labels, y=freq_labels, colorscale="Viridis")
    )

    fig.update_layout(
        title="STFT Magnitude",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Frequency"),
        coloraxis_colorbar=dict(title="Magnitude (dB)"),
    )
    # 显示图表,会在127.0.0.1随机一个端口显示
    fig.show()

if __name__=="__main__":
    import librosa
    import numpy as np
    
    filename = "audio/NIGHTFALL.mp3"
    # 加载音频文件
    y, sr = librosa.load(filename)

    # 计算STFT
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # 获取x轴和y轴的标签
    time_labels = librosa.frames_to_time(
        np.arange(stft.shape[1]), sr=sr, hop_length=hop_length
    )
    freq_labels = librosa.fft_frequencies(sr=sr, n_fft=n_fft)