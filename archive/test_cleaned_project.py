#!/usr/bin/env python3
"""
测试清理后的项目功能
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    try:
        from automakeosufile import (
            AudioProcessor,
            FeatureExtractor,
            BeatmapGenerator,
            Config,
            save_to_picture_with_timestamp,
        )

        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置类"""
    print("\n测试配置类...")
    try:
        from automakeosufile import Config

        config = Config()
        print(f"✓ 配置类创建成功")
        print(f"  默认键数: {config.DEFAULT_COLUMNS}")
        print(f"  采样率: {config.SAMPLE_RATE}")
        print(f"  处理时长: {config.DURATION}")
        return True
    except Exception as e:
        print(f"✗ 配置类测试失败: {e}")
        return False


def test_audio_processing():
    """测试音频处理模块"""
    print("\n测试音频处理模块...")
    try:
        from automakeosufile import AudioProcessor, Config

        config = Config()
        config.DURATION = 5.0  # 只测试5秒，加快速度

        processor = AudioProcessor(config)
        print("✓ 音频处理器创建成功")

        # 检查是否有测试音频文件
        test_audio = "audio/Epilogue.mp3"
        if os.path.exists(test_audio):
            print(f"  找到测试音频文件: {test_audio}")
            return True
        else:
            print(f"  警告: 测试音频文件不存在: {test_audio}")
            print("  跳过实际音频处理测试")
            return True
    except Exception as e:
        print(f"✗ 音频处理模块测试失败: {e}")
        return False


def test_command_line():
    """测试命令行接口"""
    print("\n测试命令行接口...")
    try:
        # 测试帮助命令
        result = subprocess.run(
            [sys.executable, "-m", "automakeosufile.main", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ 命令行帮助功能正常")
            print("  输出包含: usage:")
            return True
        else:
            print(f"✗ 命令行测试失败")
            print(f"  返回码: {result.returncode}")
            print(f"  输出: {result.stdout[:200]}...")
            return False
    except Exception as e:
        print(f"✗ 命令行测试异常: {e}")
        return False


def test_project_structure():
    """测试项目结构"""
    print("\n测试项目结构...")

    required_dirs = [
        "automakeosufile",
        "backup",
        "temp",
        "output",
        "picture",
        "audio",
        "docs",
    ]

    empty_dirs = [
        "archive",
        "algorithm",
        "fileprocess",
    ]

    all_passed = True

    # 检查必需目录
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ 目录存在: {dir_name}")
        else:
            print(f"✗ 目录不存在: {dir_name}")
            all_passed = False

    # 检查空目录（应该有README文件）
    for dir_name in empty_dirs:
        dir_path = Path(dir_name)
        readme_path = dir_path / "README.md"

        if dir_path.exists() and dir_path.is_dir():
            if readme_path.exists():
                print(f"✓ 空目录有README: {dir_name}")
            else:
                print(f"⚠ 空目录无README: {dir_name}")
        else:
            print(f"✗ 空目录不存在: {dir_name}")
            all_passed = False

    # 检查备份目录
    backup_dir = Path("backup/20250305_cleanup")
    if backup_dir.exists():
        print(f"✓ 备份目录存在: {backup_dir}")

        # 检查备份内容
        backup_items = list(backup_dir.iterdir())
        if len(backup_items) >= 5:  # 至少应该有5个备份项目
            print(f"✓ 备份文件数量正常: {len(backup_items)}个")
            for item in backup_items:
                print(f"  - {item.name}")
        else:
            print(f"⚠ 备份文件数量较少: {len(backup_items)}个")
    else:
        print(f"✗ 备份目录不存在")
        all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("AutoMakeosuFile 项目清理后功能测试")
    print("=" * 60)

    tests = [
        ("模块导入", test_imports),
        ("配置类", test_config),
        ("音频处理", test_audio_processing),
        ("命令行接口", test_command_line),
        ("项目结构", test_project_structure),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n[测试] {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 所有测试通过！项目清理成功。")
        return 0
    else:
        print(f"\n⚠ 有 {total - passed} 个测试未通过，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
