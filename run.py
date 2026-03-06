#!/usr/bin/env python3
"""
项目根目录的快捷运行脚本
可以方便地运行main.py进行调试
"""
import sys
import os

# 将当前目录添加到路径，以便导入algorithm模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入algorithm.main的main函数
from algorithm.main import main as algorithm_main


def main():
    """主函数 - 提供简化的命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="AutoMakeosuFile 快捷运行脚本")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="audio/Scattered Rose.wav",
        help="输入音频文件路径 (默认: audio/Scattered Rose.wav)",
    )
    parser.add_argument("--columns", type=int, default=7, help="键数 (4, 6, 7, 8)")
    parser.add_argument("--visualize", action="store_true", help="生成可视化图表")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument(
        "--process-seconds",
        type=float,
        default=20.0,
        help="处理音频的前N秒（加快测试速度）",
    )

    args = parser.parse_args()

    # 构建参数列表
    sys.argv = [
        "algorithm/main.py",
        args.audio_file,
        "--columns",
        str(args.columns),
        "--output-dir",
        args.output_dir,
    ]

    if args.visualize:
        sys.argv.append("--visualize")

    sys.argv.extend(["--process-seconds", str(args.process_seconds)])

    print("=" * 60)
    print("运行 AutoMakeosuFile")
    print(f"音频文件: {args.audio_file}")
    print(f"键数: {args.columns}K")
    print(f"输出目录: {args.output_dir}")
    print(f"处理时长: {args.process_seconds}秒")
    if args.visualize:
        print("可视化: 开启")
    print("=" * 60)

    # 调用algorithm.main.main函数
    return algorithm_main()


if __name__ == "__main__":
    sys.exit(main())
