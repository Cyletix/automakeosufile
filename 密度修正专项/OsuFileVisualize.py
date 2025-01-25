import librosa


def analyze_bpm(audio_path):
    """
    使用 librosa 分析音频的 BPM。
    """
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


class FrequencyAnalyzer:
    def __init__(self, bpm=None):
        self.bpm = bpm
        self.target_frequencies = []

    def analyze_bpm(self, audio_path):
        """
        使用 librosa 分析音频的 BPM。
        """
        y, sr = librosa.load(audio_path, sr=None)
        self.bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return self.bpm

    def calculate_frequencies(self):
        """
        根据 BPM 计算目标频率列表。
        """
        base_frequency = self.bpm / 60
        divisions = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
        self.target_frequencies = [base_frequency * div for div in divisions]
        return self.target_frequencies

    def analyze_signals(self, signals, fs=256, window_size=512, hop_size=256):
        """
        对按键信号进行频率分析。
        """
        amplitude_matrices = []
        time_axes = None
        for signal in signals:
            n_samples = len(signal)
            n_frames = (n_samples - window_size) // hop_size + 1
            time_axis = np.arange(n_frames) * hop_size / fs
            amplitude_matrix = np.zeros((len(self.target_frequencies), n_frames))

            for i, freq in enumerate(self.target_frequencies):
                omega = 2 * np.pi * freq / fs
                for frame_idx in range(n_frames):
                    start = frame_idx * hop_size
                    end = start + window_size
                    if end > n_samples:
                        break
                    windowed_signal = signal[start:end] * np.hanning(window_size)
                    amplitude_matrix[i, frame_idx] = np.abs(
                        np.sum(
                            windowed_signal
                            * np.exp(-1j * omega * np.arange(window_size))
                        )
                    )

            amplitude_matrices.append(amplitude_matrix)
            time_axes = time_axis
        return amplitude_matrices, time_axes


def calculate_custom_frequencies(bpm):
    """
    根据 BPM 计算自定义频率列表。
    bpm: 每分钟节拍数。
    返回：频率列表（单位 Hz）。
    """
    base_frequency = bpm / 60  # 1 分音对应的频率
    divisions = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
    return [base_frequency * div for div in divisions]


import numpy as np


def parse_osu_file(osu_path):
    """
    解析 .osu 文件，返回包含 (time_ms, lane, type) 的列表。
    动态从 [Difficulty] 部分读取 CircleSize，确定轨道数量。
    """
    hit_objects = []
    num_columns = None
    in_hitobjects = False

    with open(osu_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 从 [Difficulty] 部分读取 CircleSize
            if line.startswith("[Difficulty]"):
                for difficulty_line in f:
                    difficulty_line = difficulty_line.strip()
                    if difficulty_line.startswith("CircleSize:"):
                        num_columns = int(float(difficulty_line.split(":")[1].strip()))
                        break
                continue

            # 检测到 [HitObjects] 部分开始解析音符
            if line.startswith("[HitObjects]"):
                in_hitobjects = True
                continue
            if not in_hitobjects or not line:
                continue

            # 解析 HitObjects 部分的内容
            parts = line.split(",")
            if len(parts) < 5:
                continue

            try:
                x = int(parts[0])  # x 坐标决定 lane
                time_ms = int(parts[2])  # 时间戳（毫秒）
                obj_type = int(parts[3])  # 类型
            except ValueError:
                continue

            # 确保 num_columns 已从 [Difficulty] 读取
            if num_columns is None:
                raise ValueError(
                    "CircleSize (轨道数) 未从 [Difficulty] 部分正确读取，请检查文件格式！"
                )

            # 根据 x 坐标计算 lane
            lane = int(x * num_columns / 512)
            lane = max(0, min(num_columns - 1, lane))  # 确保 lane 在合法范围内

            # 添加到音符列表
            hit_objects.append((time_ms, lane, obj_type))

    # 按时间排序
    hit_objects.sort(key=lambda x: x[0])
    return hit_objects, num_columns


def custom_frequency_analysis(signal, fs, frequencies, window_size=256, hop_size=128):
    """
    针对特定频率进行离散频谱分析。
    signal: 输入信号。
    fs: 采样率。
    frequencies: 要分析的频率列表（Hz）。
    window_size: 每帧分析的窗口大小。
    hop_size: 窗口移动步长。
    返回：频率幅值矩阵，时间轴。
    """
    n_samples = len(signal)
    n_frames = (n_samples - window_size) // hop_size + 1
    time_axis = np.arange(n_frames) * hop_size / fs
    amplitude_matrix = np.zeros((len(frequencies), n_frames))

    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq / fs
        for frame_idx in range(n_frames):
            start = frame_idx * hop_size
            end = start + window_size
            if end > n_samples:
                break
            windowed_signal = signal[start:end] * np.hanning(window_size)
            amplitude_matrix[i, frame_idx] = np.abs(
                np.sum(windowed_signal * np.exp(-1j * omega * np.arange(window_size)))
            )

    return amplitude_matrix, time_axis


import matplotlib.pyplot as plt


def plot_tracks_amplitude(time_axis, frequencies, amplitude_matrices, track_labels):
    """
    修复版本：绘制多轨道频谱分析结果，每个轨道一行，时间轴共享，图例合并。
    """
    num_tracks = len(amplitude_matrices)
    fig, ax = plt.subplots(
        num_tracks, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )

    # 分别绘制每条轨道的频谱图
    for track_idx, track_amplitude in enumerate(amplitude_matrices):
        extent = [time_axis[0], time_axis[-1], frequencies[0], frequencies[-1]]
        im = ax[track_idx].imshow(
            track_amplitude,
            aspect="auto",
            extent=extent,
            origin="lower",
            cmap="viridis",
        )
        ax[track_idx].set_title(f"{track_labels[track_idx]}")
        # ax[track_idx].set_ylabel("Frequency (Hz)")

    # 添加共享时间轴
    ax[-1].set_xlabel("Time (s)")

    # 全局颜色条
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Amplitude", rotation=90)

    plt.suptitle("Frequency Analysis Across Tracks", fontsize=14)
    plt.show()


def convert_to_pulse(hit_objects, lane_id, total_length_ms=None, fs=50):
    """
    根据 lane_id 筛选音符，生成该轨道的脉冲序列。
    fs=50 => 每 20ms 一个采样点。
    """
    times = [obj[0] for obj in hit_objects if obj[1] == lane_id]
    if not times:
        return np.array([])
    if total_length_ms is None:
        total_length_ms = times[-1] + 2000  # 末尾再加 2s

    # 采样总数
    N = int(total_length_ms * fs / 1000.0)
    signal = np.zeros(N, dtype=np.float32)

    # 每个音符所在采样点 index = round(time_ms * fs / 1000)
    for t in times:
        idx = int(round(t * fs / 1000.0))
        if 0 <= idx < N:
            signal[idx] = 1.0

    return signal


def spectral_analysis(track_signals, fs=256):
    # 频谱分析
    amplitude_matrices = []
    time_axes = None
    for signal in track_signals:
        amplitude_matrix, time_axis = custom_frequency_analysis(
            signal,
            fs=fs,
            frequencies=target_frequencies,
            window_size=512,
            hop_size=256,
        )
        amplitude_matrices.append(amplitude_matrix)
        time_axes = time_axis
    return amplitude_matrices, time_axes


def plot_spectral_analysis():
    # 绘制频谱图
    plot_tracks_amplitude(
        time_axes,
        target_frequencies,
        amplitude_matrices,
        track_labels=[f"Track {i}" for i in range(num_columns)],
    )


if __name__ == "__main__":
    osu_path = r"C:\Users\Administrator\AppData\Local\osu!\Songs\2163955 Kurokotei  feat eili - Ceremony -lirile-\Kurokotei  feat. eili - Ceremony -lirile- (kojodat) [Lunar Reverie].osu.a8.osu"
    osu_audio_path = r"C:\Users\Administrator\AppData\Local\osu!\Songs\2163955 Kurokotei  feat eili - Ceremony -lirile-\audio.mp3"
    bpm = analyze_bpm(osu_audio_path)
    print(f"Detected BPM: {bpm}")

    hit_objects, num_columns = parse_osu_file(osu_path)

    # 计算目标频率
    target_frequencies = calculate_custom_frequencies(bpm)
    print(f"Target Frequencies: {target_frequencies}")

    # 示例：读取轨道信号并分析
    track_signals = [
        convert_to_pulse(hit_objects, lane_id, fs=50) for lane_id in range(num_columns)
    ]
    amplitude_matrices = []
    time_axes = None

    for signal in track_signals:
        amplitude_matrix, time_axis = custom_frequency_analysis(
            signal, fs=50, frequencies=target_frequencies
        )
        amplitude_matrices.append(amplitude_matrix)
        time_axes = time_axis

    # 绘制结果
    # 动态生成轨道标签
    track_labels = [f"Track {i}" for i in range(num_columns)]

    # 修改绘图调用
    plot_tracks_amplitude(
        time_axes,
        target_frequencies,
        amplitude_matrices,
        track_labels=track_labels,
    )


if __name__ == "__main__":
    osu_path = r"C:\Users\Administrator\AppData\Local\osu!\Songs\2163955 Kurokotei  feat eili - Ceremony -lirile-\Kurokotei  feat. eili - Ceremony -lirile- (kojodat) [Lunar Reverie].osu"
    osu_audio_path = r"C:\Users\Administrator\AppData\Local\osu!\Songs\2163955 Kurokotei  feat eili - Ceremony -lirile-\audio.mp3"

    bpm = analyze_bpm(osu_audio_path)
    print(f"Detected BPM: {bpm}")

    # 计算自定义频率
    target_frequencies = calculate_custom_frequencies(bpm)
    print(f"Custom Target Frequencies: {target_frequencies}")

    # 示例：读取轨道信号并分析
    track_signals = [
        convert_to_pulse(hit_objects, lane_id, fs=256) for lane_id in range(num_columns)
    ]

    # 频谱分析
    amplitude_matrices = []
    time_axes = None
    for signal in track_signals:
        amplitude_matrix, time_axis = custom_frequency_analysis(
            signal,
            fs=256,
            frequencies=target_frequencies,
            window_size=512,
            hop_size=256,
        )
        amplitude_matrices.append(amplitude_matrix)
        time_axes = time_axis

    # 绘制频谱图
    plot_tracks_amplitude(
        time_axes,
        target_frequencies,
        amplitude_matrices,
        track_labels=[f"Track {i}" for i in range(num_columns)],
    )
