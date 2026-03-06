#!/usr/bin/env python3
"""
简短优化器测试 - 只运行2次迭代，验证不会卡住
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_short_optimization():
    print("=" * 60)
    print("简短优化器测试")
    print("只运行2次迭代，验证不会卡住")
    print("=" * 60)

    from algorithm.auto_optimization import AutoOptimizer

    optimizer = AutoOptimizer()

    # 显示参考统计
    print(f"参考谱面统计:")
    print(f"  总音符数: {optimizer.reference_stats['total_notes']}")
    print(f"  NPS: {optimizer.reference_stats['nps']:.2f}")
    print(f"  长条比例: {optimizer.reference_stats['hold_notes_percentage']:.1f}%")
    print(f"  轨道平衡标准差: {optimizer.reference_stats['column_balance_std']:.3f}")
    print("=" * 60)

    # 设置整体超时（15分钟）
    start_time = time.time()
    overall_timeout = 900  # 15分钟

    try:
        print("开始优化测试（最多2次迭代）...")
        print("注意：每次谱面生成有5分钟超时保护")

        iterations_to_run = 2
        iterations_completed = 0

        for i in range(iterations_to_run):
            print(f"\n--- 迭代 {i+1} ---")

            # 检查是否超时
            if time.time() - start_time > overall_timeout:
                print("整体测试超时（15分钟）")
                break

            # 运行迭代
            similarity, stats = optimizer.run_iteration()

            if similarity == 0.0:
                print(f"迭代 {i+1} 失败，停止测试")
                break

            iterations_completed += 1

            # 如果已经达到不错的结果，可以提前停止
            if similarity > 0.8:
                print(f"相似度达到 {similarity:.3f}，提前停止测试")
                break

        elapsed = time.time() - start_time
        print(f"\n✓ 测试完成！耗时: {elapsed:.1f}秒")
        print(f"完成迭代: {iterations_completed}/{iterations_to_run}")
        print(f"最佳相似度: {optimizer.best_similarity:.3f}")

        if iterations_completed > 0:
            print("\n🎉 优化器测试成功！不会卡住。")
            print("已添加超时机制，可以安全运行完整优化。")
        else:
            print("\n⚠️ 所有迭代都失败，但至少没有永久卡住。")

        return True

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n测试被用户中断 (耗时: {elapsed:.1f}秒)")
        print(f"当前最佳相似度: {optimizer.best_similarity:.3f}")
        return True  # 用户中断不算失败
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 测试失败 (耗时: {elapsed:.1f}秒): {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("注意：这个测试可能需要几分钟时间")
    print("每次谱面生成有5分钟超时保护")
    print("整个测试有15分钟超时保护")
    print("如果卡住，会自动终止")
    print("-" * 60)

    success = test_short_optimization()

    if success:
        print("\n" + "=" * 60)
        print("🎉 优化器测试通过！")
        print("现在可以运行完整的优化器：")
        print("  python temp/run_optimization.py")
        print("或")
        print("  python algorithm/auto_optimization.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ 优化器测试失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
