#!/usr/bin/env python3
"""
测试优化流程 - 验证进化算法是否能正常运行
"""

import os
import sys
import json
import subprocess
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_single_iteration():
    """测试单次迭代是否能正常运行"""
    print("=" * 60)
    print("测试单次迭代")
    print("=" * 60)

    # 测试音频文件是否存在
    audio_file = "audio/Scattered Rose.mp3"
    if not os.path.exists(audio_file):
        print(f"错误: 音频文件不存在: {audio_file}")
        return False

    print(f"✓ 音频文件存在: {audio_file}")

    # 测试配置文件是否存在
    config_file = "automakeosufile/config.py"
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        return False

    print(f"✓ 配置文件存在: {config_file}")

    # 测试参考统计文件是否存在
    stats_file = "output/reference_stats.json"
    if not os.path.exists(stats_file):
        print(f"错误: 参考统计文件不存在: {stats_file}")
        return False

    print(f"✓ 参考统计文件存在: {stats_file}")

    # 创建输出目录
    output_dir = "output/optimization_experiments"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {output_dir}")

    # 测试单次谱面生成
    print("\n测试单次谱面生成...")
    cmd = [
        "python",
        "automakeosufile/main.py",
        audio_file,
        "--columns",
        "7",
        "--output-dir",
        output_dir,
        "--iteration",
        "1",
    ]

    try:
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ 谱面生成成功")

        # 检查是否生成了.osu文件
        osu_files = []
        for file in os.listdir(output_dir):
            if file.endswith("_iter1_7K.osu"):
                osu_files.append(os.path.join(output_dir, file))

        if osu_files:
            print(f"✓ 生成的.osu文件: {os.path.basename(osu_files[0])}")

            # 测试.osu文件解析
            print("\n测试.osu文件解析...")
            try:
                from temp.osu_parser import parse_osu_file

                stats = parse_osu_file(osu_files[0])
                print(f"✓ .osu文件解析成功")
                print(f"  总音符数: {stats.get('total_notes', 0)}")
                print(f"  NPS: {stats.get('nps', 0):.2f}")
                print(f"  长条比例: {stats.get('hold_notes_percentage', 0):.1f}%")
                print(f"  轨道平衡标准差: {stats.get('column_balance_std', 0):.3f}")

                # 计算相似度
                with open(stats_file, "r", encoding="utf-8") as f:
                    reference_stats = json.load(f)

                # 简单相似度计算
                note_similarity = 1.0 - min(
                    abs(stats["total_notes"] - reference_stats["total_notes"])
                    / reference_stats["total_notes"],
                    1.0,
                )
                nps_similarity = 1.0 - min(
                    abs(stats["nps"] - reference_stats["nps"]) / reference_stats["nps"],
                    1.0,
                )

                similarity = (note_similarity + nps_similarity) / 2
                print(f"  相似度估算: {similarity:.3f}")

                return True

            except Exception as e:
                print(f"✗ .osu文件解析失败: {e}")
                return False
        else:
            print("✗ 未找到生成的.osu文件")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ 谱面生成失败: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False


def test_config_update():
    """测试配置文件更新功能"""
    print("\n" + "=" * 60)
    print("测试配置文件更新")
    print("=" * 60)

    config_file = "automakeosufile/config.py"

    # 备份原始配置
    with open(config_file, "r", encoding="utf-8") as f:
        original_content = f.read()

    print("✓ 原始配置已备份")

    try:
        # 测试更新配置
        test_params = {
            "DENSITY_FILTER_RATIO": 0.7,
            "MAX_NOTES_PER_BEAT": 20,
            "HOLD_NOTE_TARGET_PERCENTAGE": 20.0,
        }

        # 读取配置
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 更新参数
        import re

        for param_name, param_value in test_params.items():
            if isinstance(param_value, float):
                search_str = f"{param_name} = "
                pattern = rf"{search_str}[0-9\.\-]+"
                replacement = f"{search_str}{param_value}"
                content = re.sub(pattern, replacement, content)
            elif isinstance(param_value, int):
                search_str = f"{param_name} = "
                pattern = rf"{search_str}[0-9\-]+"
                replacement = f"{search_str}{param_value}"
                content = re.sub(pattern, replacement, content)

        # 保存更新后的配置
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✓ 配置更新成功")

        # 验证更新
        with open(config_file, "r", encoding="utf-8") as f:
            updated_content = f.read()

        for param_name, param_value in test_params.items():
            if f"{param_name} = {param_value}" in updated_content:
                print(f"  ✓ {param_name} 已更新为 {param_value}")
            else:
                print(f"  ✗ {param_name} 更新失败")

        return True

    except Exception as e:
        print(f"✗ 配置更新失败: {e}")
        return False

    finally:
        # 恢复原始配置
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(original_content)
        print("✓ 原始配置已恢复")


def test_evolutionary_optimizer():
    """测试进化算法优化器"""
    print("\n" + "=" * 60)
    print("测试进化算法优化器")
    print("=" * 60)

    try:
        from temp.evolutionary_optimizer import EvolutionaryOptimizer

        # 创建优化器（使用最小配置以加快测试）
        optimizer = EvolutionaryOptimizer(
            population_size=3, mutation_rate=0.5  # 很小的种群  # 高变异率
        )

        print("✓ 进化算法优化器初始化成功")
        print(f"  种群大小: {optimizer.population_size}")
        print(f"  变异率: {optimizer.mutation_rate}")
        print(f"  参考谱面总音符数: {optimizer.reference_stats['total_notes']}")
        print(f"  参考谱面NPS: {optimizer.reference_stats['nps']:.2f}")

        # 测试种群初始化
        print(f"\n测试种群初始化...")
        print(f"  种群大小: {len(optimizer.population)}")

        if optimizer.population:
            individual = optimizer.population[0]
            print(f"  第一个个体参数:")
            for param, value in list(individual.items())[:5]:  # 只显示前5个参数
                print(f"    {param}: {value}")

        # 测试单次进化
        print(f"\n测试单次进化...")
        try:
            best_fitness, best_individual, best_stats = optimizer.evolve()
            print(f"  ✓ 单次进化成功")
            print(f"    最佳适应度: {best_fitness:.4f}")

            if best_individual:
                print(f"    最佳个体参数数量: {len(best_individual)}")

            return True

        except Exception as e:
            print(f"  ✗ 单次进化失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ 进化算法优化器初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("AutoMakeosuFile 优化流程测试")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # 测试1: 单次迭代
    if test_single_iteration():
        tests_passed += 1

    # 测试2: 配置文件更新
    if test_config_update():
        tests_passed += 1

    # 测试3: 进化算法优化器
    if test_evolutionary_optimizer():
        tests_passed += 1

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"通过测试: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 所有测试通过! 优化流程可以正常运行。")
        print("\n运行完整优化流程:")
        print("  python temp/evolutionary_optimizer.py")
    elif tests_passed >= 2:
        print("⚠️ 大部分测试通过，优化流程基本可用。")
        print("\n可以尝试运行优化流程:")
        print("  python temp/evolutionary_optimizer.py")
    else:
        print("❌ 测试失败较多，需要修复问题。")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
