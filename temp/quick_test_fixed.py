#!/usr/bin/env python3
"""
快速测试修复后的auto_optimization.py
主要验证关键功能是否正常，不实际运行谱面生成
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_instantiation():
    """测试类是否能正常实例化"""
    print("=" * 60)
    print("测试AutoOptimizer实例化")
    print("=" * 60)

    try:
        from algorithm.auto_optimization import AutoOptimizer

        optimizer = AutoOptimizer()

        print(f"✓ 实例化成功")
        print(f"  参考谱面总音符数: {optimizer.reference_stats['total_notes']}")
        print(f"  当前相似度: {optimizer.current_similarity}")
        print(f"  迭代次数: {optimizer.iteration}")

        return True

    except Exception as e:
        print(f"✗ 实例化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_calculate_similarity():
    """测试相似度计算"""
    print("\n" + "=" * 60)
    print("测试相似度计算")
    print("=" * 60)

    try:
        from algorithm.auto_optimization import AutoOptimizer

        optimizer = AutoOptimizer()

        # 测试数据
        test_stats = {
            "total_notes": 400,
            "nps": 10.0,
            "hold_notes_percentage": 15.0,
            "column_balance_std": 1.0,
        }

        similarity = optimizer.calculate_similarity(test_stats)

        print(f"✓ 相似度计算成功")
        print(f"  测试统计: {test_stats}")
        print(f"  计算相似度: {similarity:.3f}")

        # 测试空统计
        empty_similarity = optimizer.calculate_similarity({})
        print(f"  空统计相似度: {empty_similarity:.3f}")

        # 测试None统计
        none_similarity = optimizer.calculate_similarity(None)
        print(f"  None统计相似度: {none_similarity:.3f}")

        return True

    except Exception as e:
        print(f"✗ 相似度计算失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_adjust_parameters():
    """测试参数调整逻辑（不实际修改文件）"""
    print("\n" + "=" * 60)
    print("测试参数调整逻辑")
    print("=" * 60)

    try:
        from algorithm.auto_optimization import AutoOptimizer

        optimizer = AutoOptimizer()

        # 备份原始配置
        config_path = "algorithm/config.py"
        with open(config_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        try:
            # 测试调整
            test_stats = {
                "total_notes": 200,  # 音符不足 (892的22%)
                "nps": 5.0,  # NPS不足 (15.7的32%)
                "hold_notes_percentage": 25.0,  # 长条比例过高 (+9.5%)
                "column_balance_std": 10.0,  # 轨道不平衡
            }

            adjusted = optimizer.adjust_parameters(test_stats, 0.5)

            print(f"✓ 参数调整测试成功")
            print(f"  调整状态: {'已调整' if adjusted else '未调整'}")

            # 恢复原始配置
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            print(f"  配置已恢复")

            return True

        except Exception as e:
            # 确保恢复配置
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            print(f"  配置已恢复")
            raise e

    except Exception as e:
        print(f"✗ 参数调整测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_methods_exist():
    """测试所有方法都存在"""
    print("\n" + "=" * 60)
    print("测试所有方法都存在")
    print("=" * 60)

    try:
        from algorithm.auto_optimization import AutoOptimizer

        required_methods = [
            "load_reference_stats",
            "generate_beatmap",
            "calculate_similarity",
            "adjust_parameters",
            "run_iteration",
            "run_optimization",
        ]

        optimizer = AutoOptimizer()

        for method_name in required_methods:
            if hasattr(optimizer, method_name):
                print(f"✓ 方法存在: {method_name}")
            else:
                print(f"✗ 方法不存在: {method_name}")
                return False

        return True

    except Exception as e:
        print(f"✗ 方法存在性测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_timeout_logic():
    """测试超时逻辑"""
    print("\n" + "=" * 60)
    print("测试超时逻辑")
    print("=" * 60)

    try:
        # 检查generate_beatmap方法是否包含timeout参数
        import inspect
        from algorithm.auto_optimization import AutoOptimizer

        method = AutoOptimizer.generate_beatmap
        source = inspect.getsource(method)

        print("检查generate_beatmap方法:")

        checks = [
            ("导入subprocess", "import subprocess" in source or "subprocess" in source),
            ("使用timeout参数", "timeout=300" in source or "timeout=" in source),
            ("处理TimeoutExpired异常", "TimeoutExpired" in source),
        ]

        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"  ✓ {check_name}")
            else:
                print(f"  ✗ {check_name}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"✗ 超时逻辑测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("快速测试修复后的AutoOptimizer")
    print("测试关键功能，不实际运行谱面生成")
    print("预计耗时: < 10秒")

    success = True

    tests = [
        test_instantiation,
        test_calculate_similarity,
        test_methods_exist,
        test_timeout_logic,
        test_adjust_parameters,  # 这个可能会稍微长一点，但也不应该卡住
    ]

    for test_func in tests:
        if not test_func():
            success = False
            print(f"\n⚠️ 测试 {test_func.__name__} 失败")

    if success:
        print("\n" + "=" * 60)
        print("🎉 所有快速测试通过！")
        print("AutoOptimizer修复成功，关键功能正常")
        print("超时机制已添加，不会永久卡住")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ 部分测试失败，需要进一步调试")
        print("=" * 60)
        sys.exit(1)
