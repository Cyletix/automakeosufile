#!/usr/bin/env python3
"""
测试优化器基本功能
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.auto_optimization import AutoOptimizer
from algorithm.evolutionary_optimizer import EvolutionaryOptimizer


def test_auto_optimizer():
    print("=== 测试 AutoOptimizer ===")
    optimizer = AutoOptimizer()
    print(f"参考谱面总音符数: {optimizer.reference_stats['total_notes']}")
    print(f"参考谱面NPS: {optimizer.reference_stats['nps']}")
    print(f"参考谱面长条比例: {optimizer.reference_stats['hold_notes_percentage']}%")
    print(f"参考谱面轨道平衡标准差: {optimizer.reference_stats['column_balance_std']}")

    # 测试一次迭代
    print("\n--- 测试单次迭代 ---")
    similarity, stats = optimizer.run_iteration()
    print(f"相似度: {similarity}")
    print(f"统计: {stats}")

    print("\n✓ AutoOptimizer 基本功能正常")


def test_evolutionary_optimizer():
    print("\n=== 测试 EvolutionaryOptimizer ===")
    optimizer = EvolutionaryOptimizer(population_size=2, mutation_rate=0.3)
    print(f"种群大小: {optimizer.population_size}")
    print(f"参考谱面统计已加载")

    # 测试初始化种群
    print(f"种群初始化完成，大小: {len(optimizer.population)}")

    # 测试更新配置
    test_individual = optimizer.population[0]
    print(f"测试更新配置...")
    optimizer.update_config_file(test_individual, 1)
    print("✓ 配置更新成功")

    print("\n✓ EvolutionaryOptimizer 基本功能正常")


if __name__ == "__main__":
    try:
        test_auto_optimizer()
        test_evolutionary_optimizer()
        print("\n🎉 所有优化器基本功能测试通过!")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
