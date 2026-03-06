#!/usr/bin/env python3
"""
分析项目中的Python文件使用情况
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List


def get_all_python_files(root_dir: str) -> List[str]:
    """获取所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过一些目录
        if "__pycache__" in root or ".git" in root:
            continue

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files


def extract_imports(file_path: str) -> Set[str]:
    """提取Python文件中的导入语句"""
    imports = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")

    return imports


def analyze_file_usage(root_dir: str):
    """分析文件使用情况"""
    print("=" * 80)
    print("AutoMakeosuFile 项目文件使用情况分析")
    print("=" * 80)

    # 获取所有Python文件
    all_files = get_all_python_files(root_dir)
    print(f"\n找到 {len(all_files)} 个Python文件")

    # 获取主入口文件
    main_files = []
    for file in all_files:
        filename = os.path.basename(file)
        if filename == "main.py" or filename == "main_old.py":
            main_files.append(file)

    print(f"\n主入口文件: {len(main_files)} 个")
    for main_file in main_files:
        print(f"  - {os.path.relpath(main_file, root_dir)}")

    # 分析automakeosufile模块的使用
    print("\n" + "=" * 80)
    print("automakeosufile 模块分析")
    print("=" * 80)

    automake_dir = os.path.join(root_dir, "automakeosufile")
    automake_files = get_all_python_files(automake_dir)

    print(f"\nautomakeosufile 模块包含 {len(automake_files)} 个文件:")
    for file in automake_files:
        rel_path = os.path.relpath(file, root_dir)
        print(f"  - {rel_path}")

    # 检查主文件中的导入
    print("\n" + "=" * 80)
    print("主文件导入分析")
    print("=" * 80)

    for main_file in main_files:
        print(f"\n分析: {os.path.relpath(main_file, root_dir)}")
        imports = extract_imports(main_file)

        # 检查是否导入automakeosufile模块
        automake_imports = [imp for imp in imports if "automakeosufile" in imp]
        if automake_imports:
            print(f"  导入的automakeosufile模块: {automake_imports}")
        else:
            print(f"  未导入automakeosufile模块")

        # 检查是否导入其他本地模块
        local_imports = []
        for imp in imports:
            # 检查是否是本地文件导入
            if any(
                keyword in imp
                for keyword in ["algorithm", "fileprocess", "密度修正专项"]
            ):
                local_imports.append(imp)

        if local_imports:
            print(f"  导入的其他本地模块: {local_imports}")

    # 分析重复功能
    print("\n" + "=" * 80)
    print("功能重复分析")
    print("=" * 80)

    # 检查OSU文件处理功能
    print("\n1. OSU文件处理功能:")
    osu_parsers = []
    for file in all_files:
        filename = os.path.basename(file)
        if "osu" in filename.lower() and "parser" in filename.lower():
            osu_parsers.append(file)
        elif "osu" in filename.lower() and "file" in filename.lower():
            osu_parsers.append(file)

    print(f"  找到 {len(osu_parsers)} 个OSU文件处理相关文件:")
    for parser in osu_parsers:
        rel_path = os.path.relpath(parser, root_dir)
        print(f"    - {rel_path}")

    # 检查二值化功能
    print("\n2. 二值化功能:")
    binarize_files = []
    for file in all_files:
        filename = os.path.basename(file)
        if "binarize" in filename.lower():
            binarize_files.append(file)

    print(f"  找到 {len(binarize_files)} 个二值化相关文件:")
    for bfile in binarize_files:
        rel_path = os.path.relpath(bfile, root_dir)
        print(f"    - {rel_path}")

    # 检查音频处理功能
    print("\n3. 音频处理功能:")
    audio_files = []
    for file in all_files:
        filename = os.path.basename(file)
        if "audio" in filename.lower() and "processing" in filename.lower():
            audio_files.append(file)
        elif "feature" in filename.lower() and "extract" in filename.lower():
            audio_files.append(file)

    print(f"  找到 {len(audio_files)} 个音频处理相关文件:")
    for afile in audio_files:
        rel_path = os.path.relpath(afile, root_dir)
        print(f"    - {rel_path}")

    # 检查未使用的文件
    print("\n" + "=" * 80)
    print("可能未使用的文件分析")
    print("=" * 80)

    # 检查archive目录
    archive_files = get_all_python_files(os.path.join(root_dir, "archive"))
    print(f"\narchive目录中有 {len(archive_files)} 个文件，这些可能是旧版本或实验代码")

    # 检查algorithm目录
    algorithm_files = get_all_python_files(os.path.join(root_dir, "algorithm"))
    print(f"\nalgorithm目录中有 {len(algorithm_files)} 个文件")

    # 检查fileprocess目录
    fileprocess_files = get_all_python_files(os.path.join(root_dir, "fileprocess"))
    print(f"\nfileprocess目录中有 {len(fileprocess_files)} 个文件")

    # 检查密度修正专项目录
    density_files = get_all_python_files(os.path.join(root_dir, "密度修正专项"))
    print(f"\n密度修正专项目录中有 {len(density_files)} 个文件")

    print("\n" + "=" * 80)
    print("建议:")
    print("=" * 80)
    print(
        """
1. 项目存在明显的功能重复，特别是OSU文件处理、二值化等功能
2. automakeosufile模块是当前的主要实现，其他目录中的文件可能是旧版本或实验代码
3. archive目录中的文件可以安全地归档或删除
4. 建议统一使用automakeosufile模块，清理重复的代码
5. 需要建立清晰的模块依赖关系，避免重复造轮子
    """
    )


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analyze_file_usage(root_dir)
