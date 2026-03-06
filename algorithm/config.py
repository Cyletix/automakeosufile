"""
优化配置文件 v2 - 更激进的参数调整
目标: 使生成的谱面接近 Scattered Rose.osu 的统计指标
"""

from typing import Optional


class Config:
    # 音频处理参数
    SAMPLE_RATE = 22050
    DURATION: Optional[float] = 20.0  # 只处理前20秒进行算法测试
    MONO = True

    # STFT参数
    N_FFT = 2048
    HOP_LENGTH = 512

    # Mel频谱参数
    N_MELS = 128
    FMIN = 20
    FMAX = 8000

    # 二值化参数 - 极低阈值，最大化检测
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 15  # 更小的块大小，提高局部灵敏度
    ADAPTIVE_THRESHOLD_C = -8  # 负阈值，检测所有可能的音符
    MORPH_KERNEL_SIZE = 2  # 极小的核大小，保留所有细节

    # 音符检测参数
    MIN_NOTE_DURATION_MS = 5  # 极低最小持续时间
    MAX_NOTE_DURATION_MS = 2000  # 最大音符持续时间(ms)
    NOTE_GAP_MS = 10  # 极小的最小间隔

    # 节拍对齐参数
    BEAT_DIVISORS = [1, 2, 4, 8, 16, 32]  # 允许所有节拍细分，包括32分音
    MAX_ALIGN_ERROR_MS = 100  # 非常宽松的对齐误差

    # 物理手感参数
    MIN_COLUMN_GAP_MS = 10  # 极小的同一轨道最小间隔

    # 谱面生成参数
    DEFAULT_COLUMNS = 7  # 默认7K
    COLUMN_MAPPING = {
        4: [0, 3, 7, 10],  # 4K: C, E, G, B
        6: [0, 2, 4, 5, 7, 9],  # 6K: C, D, E, F, G, A
        7: list(range(7)),  # 7K: 使用前7个音高
        8: list(range(8)),  # 8K: 使用前8个音高
    }

    # 密度控制参数
    MAX_NOTES_PER_BEAT = 16  # 每拍最大16个音符（非常密集）
    MAX_SAME_COLUMN_INTERVAL_MS = 50  # 同一轨道最小间隔(ms)

    # 能量平滑窗口
    ENERGY_SMOOTHING_WINDOW = 20

    # 密度映射 (NPS值大幅提高，接近目标15.7)
    DENSITY_MAPPING = [
        (0.0, 20.0),  # 极低能量：NPS=20.0
        (0.2, 22.0),  # 低能量：NPS=22.0
        (0.4, 24.0),  # 中等能量：NPS=24.0
        (0.6, 26.0),  # 高能量：NPS=26.0
        (0.8, 28.0),  # 极高能量：NPS=28.0
    ]

    # 节拍细分限制：完全放开限制
    SNAP_RESTRICTIONS = [
        (0.0, [1, 2, 4, 8, 16, 32]),  # 所有能量级别都允许所有细分
        (0.2, [1, 2, 4, 8, 16, 32]),
        (0.4, [1, 2, 4, 8, 16, 32]),
        (0.6, [1, 2, 4, 8, 16, 32]),
        (0.8, [1, 2, 4, 8, 16, 32]),
    ]

    # 输出参数
    OUTPUT_DIR = "output"
    VISUALIZE = True

    # 长条生成参数 - 大幅减少长条比例
    HOLD_NOTE_MIN_DURATION = 300  # 提高最小持续时间，减少长条数量
    HOLD_NOTE_MAX_DURATION = 800  # 降低最大长条持续时间
    HOLD_NOTE_TARGET_PERCENTAGE = 15  # 目标长条比例15%

    # 轨道平衡参数
    COLUMN_BALANCE_TARGET_STD = 2.0  # 目标轨道平衡标准差
    COLUMN_REBALANCE_THRESHOLD = 0.2  # 轨道重新平衡阈值

    # 密度控制调整
    DENSITY_FILTER_RATIO = 0.5  # 保留30%的音符（之前是7.1%）

    # 物理手感修正调整
    PHYSICAL_CORRECTION_STRICTNESS = 0.5  # 降低物理修正严格度
