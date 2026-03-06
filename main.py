#!/usr/bin/env python3
"""
AutoMakeosuFile - 自动生成 osu!mania 谱面
根目录入口文件，支持直接 F5 调试
"""

import sys
import os

# 添加 algorithm 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithm.main import main

if __name__ == "__main__":
    sys.exit(main())
