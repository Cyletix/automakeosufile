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
