"""
主程序 - 使用新的模块化结构生成谱面
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automakeosufile import AudioProcessor, FeatureExtractor, BeatmapGenerator, Config


def visualize_results(audio_data, features, output_dir="output"):
    """
    可视化处理结果
    """
    print("生成可视化结果...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. 原始Mel频谱
    log_mel = audio_data["log_mel"]
    im1 = axes[0].imshow(log_mel, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Mel频谱图 (dB)")
    axes[0].set_ylabel("频率bin")
    plt.colorbar(im1, ax=axes[0])

    # 2. 二值化结果
    binary_matrix = audio_data["binary_matrix"]
    im2 = axes[1].imshow(binary_matrix, aspect="auto", origin="lower", cmap="gray")
    axes[1].set_title("自适应二值化结果")
    axes[1].set_ylabel("频率bin")

    # 3. 音符分布
    controlled_notes = features["controlled_notes"]
    if controlled_notes:
        times = [n["aligned_time"] / 1000 for n in controlled_notes]  # 转换为秒
        columns = [n["column"] for n in controlled_notes]

        axes[2].scatter(times, columns, alpha=0.6, s=10)
        axes[2].set_title("音符分布 (时间 vs 轨道)")
        axes[2].set_xlabel("时间 (秒)")
        axes[2].set_ylabel("轨道")
        axes[2].set_ylim(-0.5, features["config"]["columns"] - 0.5)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "processing_results.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"可视化结果保存到: {output_path}")

    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"音频时长: {len(audio_data['audio']) / audio_data['sample_rate']:.1f}秒")
    print(f"检测到BPM: {features['bpm_info']['bpm']:.1f}")
    print(f"提取音符事件: {len(audio_data['note_events'])}个")
    print(f"节拍对齐后: {len(features['aligned_notes'])}个")
    print(f"密度控制后: {len(features['controlled_notes'])}个")

    # 轨道分布统计
    if controlled_notes:
        column_counts = {}
        for note in controlled_notes:
            col = note["column"]
            column_counts[col] = column_counts.get(col, 0) + 1

        print("\n轨道分布:")
        for col in sorted(column_counts.keys()):
            print(
                f"  轨道{col}: {column_counts[col]}个音符 ({column_counts[col]/len(controlled_notes):.1%})"
            )


def main():
    parser = argparse.ArgumentParser(description="自动生成osu!mania谱面")
    parser.add_argument("audio_file", help="输入音频文件路径 (mp3或wav)")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--columns", type=int, default=7, help="键数 (4, 6, 7, 8)")
    parser.add_argument("--visualize", action="store_true", help="生成可视化图表")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.audio_file):
        print(f"错误: 文件不存在: {args.audio_file}")
        return 1

    # 创建配置
    config = Config()
    config.DEFAULT_COLUMNS = args.columns
    config.OUTPUT_DIR = args.output_dir
    config.VISUALIZE = args.visualize

    try:
        print("=" * 60)
        print("AutoMakeosuFile v2.0 - 改进版谱面生成器")
        print("=" * 60)

        # 1. 音频处理
        print("\n[阶段1] 音频处理")
        audio_processor = AudioProcessor(config)
        audio_data = audio_processor.process_audio(args.audio_file)

        # 2. 特征提取
        print("\n[阶段2] 特征提取")
        feature_extractor = FeatureExtractor(config)
        features = feature_extractor.extract_features(
            audio_data, audio_data["note_events"]
        )

        # 3. 谱面生成
        print("\n[阶段3] 谱面生成")
        beatmap_generator = BeatmapGenerator(config)
        output_path = beatmap_generator.generate_beatmap(
            args.audio_file, features, args.output_dir
        )

        # 4. 可视化
        if args.visualize:
            visualize_results(audio_data, features, args.output_dir)

        print("\n" + "=" * 60)
        print("✓ 谱面生成完成!")
        print(f"输出文件: {output_path}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
