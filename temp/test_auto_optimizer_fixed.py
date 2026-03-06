#!/usr/bin/env python3
"""
测试修复后的auto_optimization.py
只运行一次迭代，验证不会卡住
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_single_iteration():
    """测试单次迭代，设置更短的超时"""
    from algorithm.auto_optimization import AutoOptimizer

    print("=" * 60)
    print("测试修复后的AutoOptimizer")
    print("只运行一次迭代，验证不会卡住")
    print("=" * 60)

    optimizer = AutoOptimizer()

    # 显示参考统计
    print(f"参考谱面统计:")
    print(f"  总音符数: {optimizer.reference_stats['total_notes']}")
    print(f"  NPS: {optimizer.reference_stats['nps']:.2f}")
    print(f"  长条比例: {optimizer.reference_stats['hold_notes_percentage']:.1f}%")
    print(f"  轨道平衡标准差: {optimizer.reference_stats['column_balance_std']:.3f}")
    print("=" * 60)

    # 设置整体超时（10分钟）
    start_time = time.time()
    timeout_seconds = 600  # 10分钟

    try:
        print("开始第一次迭代...")
        print("注意：谱面生成有5分钟超时保护")

        # 运行一次迭代
        similarity, stats = optimizer.run_iteration()

        elapsed = time.time() - start_time
        print(f"\n✓ 迭代完成！耗时: {elapsed:.1f}秒")
        print(f"相似度: {similarity:.3f}")

        if stats:
            total_notes = stats.get("total_notes", 0)
            nps = stats.get("nps", 0.0)
            hold_percentage = stats.get("hold_percentage") or stats.get(
                "hold_notes_percentage", 0.0
            )
            column_std = stats.get("column_std") or stats.get("column_balance_std", 0.0)

            print(f"生成谱面统计:")
            print(f"  总音符数: {total_notes}")
            print(f"  NPS: {nps:.2f}")
            print(f"  长条比例: {hold_percentage:.1f}%")
            print(f"  轨道平衡标准差: {column_std:.2f}")

        print("\n🎉 AutoOptimizer修复成功！不会卡住。")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 测试失败 (耗时: {elapsed:.1f}秒): {e}")
        import traceback

        traceback.print_exc()
        return False


def test_generate_beatmap_directly():
    """直接测试generate_beatmap方法"""
    print("\n" + "=" * 60)
    print("直接测试generate_beatmap方法")
    print("=" * 60)

    from algorithm.auto_optimization import AutoOptimizer

    optimizer = AutoOptimizer()

    start_time = time.time()
    try:
        # 直接调用generate_beatmap，设置更短的超时
        print("调用generate_beatmap...")
        stats = optimizer.generate_beatmap()

        elapsed = time.time() - start_time
        if stats:
            print(f"✓ generate_beatmap成功！耗时: {elapsed:.1f}秒")
            print(f"返回统计: {stats}")
            return True
        else:
            print(f"✗ generate_beatmap返回空统计 (耗时: {elapsed:.1f}秒)")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ generate_beatmap异常 (耗时: {elapsed:.1f}秒): {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("注意：测试可能需要几分钟时间，因为有5分钟超时保护")
    print("如果卡住，5分钟后会自动终止")

    success = True

    # 先测试直接生成谱面
    if not test_generate_beatmap_directly():
        success = False

    # 然后测试完整迭代
    if not test_single_iteration():
        success = False

    if success:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！AutoOptimizer修复成功")
        print("现在可以正常运行优化器，不会永久卡住")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  测试失败，需要进一步调试")
        print("=" * 60)
        sys.exit(1)
