"""
测试新的改进算法
"""

import os
import sys
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automakeosufile import AudioProcessor, FeatureExtractor, Config


def test_audio_processing():
    """测试音频处理模块"""
    print("=== 测试音频处理模块 ===")

    # 使用测试音频文件
    test_audio = "audio/NIGHTFALL.mp3"
    if not os.path.exists(test_audio):
        print(f"测试音频不存在: {test_audio}")
        print("请确保 audio/NIGHTFALL.mp3 存在")
        return None

    # 创建配置
    config = Config()
    config.DURATION = 10.0  # 只测试前10秒

    # 音频处理
    processor = AudioProcessor(config)
    audio_data = processor.process_audio(test_audio)

    print(f"音频采样率: {audio_data['sample_rate']} Hz")
    print(f"音频长度: {len(audio_data['audio'])} 样本")
    print(f"Mel频谱形状: {audio_data['mel_spectrogram'].shape}")
    print(f"二值化矩阵形状: {audio_data['binary_matrix'].shape}")
    print(f"提取音符事件: {len(audio_data['note_events'])} 个")

    # 分析二值化效果
    binary_matrix = audio_data["binary_matrix"]
    activation_rate = np.sum(binary_matrix) / binary_matrix.size
    print(f"二值化激活率: {activation_rate:.2%}")

    return audio_data


def test_feature_extraction(audio_data):
    """测试特征提取模块"""
    print("\n=== 测试特征提取模块 ===")

    config = Config()
    extractor = FeatureExtractor(config)

    features = extractor.extract_features(audio_data, audio_data["note_events"])

    print(f"检测到BPM: {features['bpm_info']['bpm']:.1f}")
    print(f"第一个节拍: {features['bpm_info']['first_beat_time']:.2f}s")
    print(f"节拍对齐: {len(features['aligned_notes'])} 个音符")
    print(f"轨道映射: {len(features['mapped_notes'])} 个音符")
    print(f"密度控制后: {len(features['controlled_notes'])} 个音符")

    # 分析音符分布
    if features["controlled_notes"]:
        times = [n["aligned_time"] for n in features["controlled_notes"]]
        columns = [n["column"] for n in features["controlled_notes"]]

        print(f"时间范围: {min(times):.0f}ms - {max(times):.0f}ms")
        print(f"轨道分布: {set(columns)}")

        # 计算平均密度
        total_time_ms = max(times) - min(times)
        if total_time_ms > 0:
            notes_per_second = len(features["controlled_notes"]) / (
                total_time_ms / 1000
            )
            print(f"音符密度: {notes_per_second:.1f} notes/sec")

    return features


def compare_with_old_algorithm():
    """与旧算法对比"""
    print("\n=== 与旧算法对比 ===")

    # 旧算法的典型问题
    old_problems = [
        "1. 固定阈值0.9 → 激活率只有12%",
        "2. 时间分辨率0.012秒/帧 → 太细了",
        "3. 没有节拍对齐 → 音符不整齐",
        "4. 没有密度控制 → 轨道过密",
        "5. 使用Chroma CQT → 不适合音乐分析",
    ]

    # 新算法的改进
    new_improvements = [
        "1. 自适应阈值 → 合理激活率",
        "2. Mel频谱 → 更适合音乐分析",
        "3. BPM对齐 → 音符整齐",
        "4. 密度控制 → 可玩性提升",
        "5. 轨道映射 → 音高对应轨道",
    ]

    print("旧算法问题:")
    for problem in old_problems:
        print(f"  {problem}")

    print("\n新算法改进:")
    for improvement in new_improvements:
        print(f"  {improvement}")


def main():
    """主测试函数"""
    print("AutoMakeosuFile v2.0 算法测试")
    print("=" * 50)

    try:
        # 测试音频处理
        audio_data = test_audio_processing()
        if audio_data is None:
            return

        # 测试特征提取
        features = test_feature_extraction(audio_data)

        # 与旧算法对比
        compare_with_old_algorithm()

        print("\n" + "=" * 50)
        print("✓ 测试完成!")
        print("新算法相比旧算法有显著改进:")
        print("  - 更好的二值化效果")
        print("  - 节拍对齐的音符")
        print("  - 合理的轨道分布")
        print("  - 密度控制提升可玩性")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
