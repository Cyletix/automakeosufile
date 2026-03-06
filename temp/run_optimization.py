#!/usr/bin/env python3
"""
简单优化器启动脚本
运行 auto_optimization 进行自动优化
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.auto_optimization import AutoOptimizer


def main():
    print("=" * 60)
    print("AutoMakeosuFile 自动优化器启动")
    print("=" * 60)
    print("使用规则优化器 (auto_optimization.py)")
    print("目标：提升谱面相似度到 0.9")
    print("参考谱面：Scattered Rose.osu")
    print("=" * 60)

    optimizer = AutoOptimizer()

    # 显示参考统计
    print(f"参考谱面统计:")
    print(f"  总音符数: {optimizer.reference_stats['total_notes']}")
    print(f"  NPS: {optimizer.reference_stats['nps']:.2f}")
    print(f"  长条比例: {optimizer.reference_stats['hold_notes_percentage']:.1f}%")
    print(f"  轨道平衡标准差: {optimizer.reference_stats['column_balance_std']:.3f}")
    print("=" * 60)

    try:
        # 运行优化
        best_similarity, best_stats = optimizer.run_optimization(
            target_similarity=0.85,  # 目标相似度85%（更容易达到）
            max_iterations=20,  # 最多20次迭代
        )

        print(f"\n优化完成!")
        print(f"最佳相似度: {best_similarity:.3f}")

        if best_similarity >= 0.85:
            print("🎉 达到目标相似度!")
        else:
            print(f"未达到目标相似度，但找到了最佳配置")

        # 保存结果
        output_dir = "output/optimization_experiments"
        os.makedirs(output_dir, exist_ok=True)

        result_file = os.path.join(output_dir, "optimization_result.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("AutoOptimization 结果\n")
            f.write("=" * 40 + "\n")
            f.write(f"最佳相似度: {best_similarity:.3f}\n")
            f.write(f"参考谱面音符数: {optimizer.reference_stats['total_notes']}\n")
            if best_stats:
                f.write(f"最佳配置音符数: {best_stats.get('total_notes', 'N/A')}\n")
                f.write(f"最佳配置NPS: {best_stats.get('nps', 'N/A')}\n")
            f.write(f"迭代次数: {optimizer.iteration}\n")

        print(f"结果保存到: {result_file}")

    except KeyboardInterrupt:
        print("\n\n优化被用户中断")
        print(f"当前最佳相似度: {optimizer.best_similarity:.3f}")
    except Exception as e:
        print(f"\n优化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
