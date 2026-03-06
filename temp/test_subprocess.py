#!/usr/bin/env python3
"""
测试子进程调用，找出卡住的原因
"""
import subprocess
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_without_quotes():
    """测试不带引号（原始方式）"""
    print("测试1: 不带引号的命令")
    cmd = [
        "python",
        "algorithm/main.py",
        "audio/Scattered Rose.mp3",  # 有空格的文件名
        "--columns",
        "7",
        "--output-dir",
        "output/optimization_experiments",
    ]

    print(f"命令: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        print(f"成功! 耗时: {elapsed:.1f}秒")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出前100字符: {result.stdout[:100]}")
        if result.stderr:
            print(f"错误: {result.stderr[:200]}")
        return True
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"超时! 耗时: {elapsed:.1f}秒")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"异常: {e} (耗时: {elapsed:.1f}秒)")
        return False


def test_with_quotes_in_shell():
    """测试在shell中使用引号"""
    print("\n测试2: 在shell中使用引号")
    cmd = 'python algorithm/main.py "audio/Scattered Rose.mp3" --columns 7 --output-dir output/optimization_experiments'

    print(f"命令: {cmd}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, shell=True, timeout=60
        )
        elapsed = time.time() - start
        print(f"成功! 耗时: {elapsed:.1f}秒")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出前100字符: {result.stdout[:100]}")
        if result.stderr:
            print(f"错误: {result.stderr[:200]}")
        return True
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"超时! 耗时: {elapsed:.1f}秒")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"异常: {e} (耗时: {elapsed:.1f}秒)")
        return False


def test_short_audio():
    """测试处理音频前20秒（更快）"""
    print("\n测试3: 使用--process-seconds参数限制处理时间")
    cmd = [
        "python",
        "algorithm/main.py",
        "audio/Scattered Rose.mp3",
        "--columns",
        "7",
        "--output-dir",
        "output/optimization_experiments",
        "--process-seconds",
        "20",  # 只处理前20秒
    ]

    print(f"命令: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        print(f"成功! 耗时: {elapsed:.1f}秒")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出前100字符: {result.stdout[:100]}")
        if result.stderr:
            print(f"错误: {result.stderr[:200]}")
        return True
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"超时! 耗时: {elapsed:.1f}秒")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"异常: {e} (耗时: {elapsed:.1f}秒)")
        return False


def test_wav_file():
    """测试使用.wav文件（不需要转换）"""
    print("\n测试4: 使用.wav文件（应该更快）")
    cmd = [
        "python",
        "algorithm/main.py",
        "audio/Scattered Rose.wav",  # 使用.wav文件
        "--columns",
        "7",
        "--output-dir",
        "output/optimization_experiments",
    ]

    print(f"命令: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        print(f"成功! 耗时: {elapsed:.1f}秒")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出前100字符: {result.stdout[:100]}")
        if result.stderr:
            print(f"错误: {result.stderr[:200]}")
        return True
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"超时! 耗时: {elapsed:.1f}秒")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"异常: {e} (耗时: {elapsed:.1f}秒)")
        return False


if __name__ == "__main__":
    print("测试子进程调用，找出卡住的原因")
    print("=" * 60)

    # 创建输出目录
    os.makedirs("output/optimization_experiments", exist_ok=True)

    tests = [
        test_without_quotes,
        test_with_quotes_in_shell,
        test_short_audio,
        test_wav_file,
    ]

    results = []
    for test_func in tests:
        results.append(test_func())
        print("-" * 60)

    print("测试结果汇总:")
    for i, (test_func, success) in enumerate(zip(tests, results), 1):
        print(f"  测试{i}: {test_func.__name__} - {'通过' if success else '失败'}")

    if any(results):
        print("\n✓ 至少有一种方法可以正常工作")
    else:
        print("\n✗ 所有方法都失败")
