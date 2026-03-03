"""
配置文件 - 包含所有可调参数
"""

from typing import Optional


class Config:
    # 音频处理参数
    SAMPLE_RATE = 22050
    DURATION: Optional[float] = None  # None表示加载完整音频
    MONO = True

    # STFT参数
    N_FFT = 2048
    HOP_LENGTH = 512

    # Mel频谱参数
    N_MELS = 128
    FMIN = 20
    FMAX = 8000

    # 二值化参数
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
    ADAPTIVE_THRESHOLD_C = 2
    MORPH_KERNEL_SIZE = 3

    # 音符检测参数
    MIN_NOTE_DURATION_MS = 50  # 最小音符持续时间(ms)
    MAX_NOTE_DURATION_MS = 500  # 最大音符持续时间(ms)
    NOTE_GAP_MS = 30  # 音符间最小间隔(ms)

    # 谱面生成参数
    DEFAULT_COLUMNS = 7  # 默认7K
    COLUMN_MAPPING = {
        4: [0, 3, 7, 10],  # 4K: C, E, G, B
        6: [0, 2, 4, 5, 7, 9],  # 6K: C, D, E, F, G, A
        7: list(range(7)),  # 7K: 使用前7个音高
        8: list(range(8)),  # 8K: 使用前8个音高
    }

    # 密度控制参数
    MAX_NOTES_PER_BEAT = 4  # 每拍最大音符数
    MAX_SAME_COLUMN_INTERVAL_MS = 100  # 同一轨道最小间隔(ms)

    # 输出参数
    OUTPUT_DIR = "output"
    VISUALIZE = True
