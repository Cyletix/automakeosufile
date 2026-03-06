"""
工具函数模块 - 处理文件操作、时间戳等
"""

import os
import datetime
import shutil


def add_timestamp_to_filename(filename):
    """
    为文件名添加时间戳

    参数:
        filename: 原始文件名 (如 "analysis.txt")

    返回:
        带时间戳的文件名 (如 "analysis_20240304_123456.txt")
    """
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 组合新文件名
    return f"{name}_{timestamp}{ext}"


def save_to_temp_with_timestamp(data, filename, temp_dir="temp"):
    """
    保存数据到临时文件，文件名添加时间戳

    参数:
        data: 要保存的数据
        filename: 原始文件名
        temp_dir: 临时目录

    返回:
        保存的文件路径
    """
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)

    # 添加时间戳
    timestamped_filename = add_timestamp_to_filename(filename)
    filepath = os.path.join(temp_dir, timestamped_filename)

    # 保存数据
    if isinstance(data, str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
    elif isinstance(data, dict):
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif isinstance(data, bytes):
        with open(filepath, "wb") as f:
            f.write(data)
    else:
        # 尝试使用pickle保存
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    print(f"临时文件已保存: {filepath}")
    return filepath


def save_to_picture_with_timestamp(fig, filename, picture_dir="picture"):
    """
    保存图片到picture文件夹，文件名添加时间戳

    参数:
        fig: matplotlib图形对象
        filename: 原始文件名
        picture_dir: 图片目录

    返回:
        保存的文件路径
    """
    # 创建picture目录
    os.makedirs(picture_dir, exist_ok=True)

    # 添加时间戳
    timestamped_filename = add_timestamp_to_filename(filename)
    filepath = os.path.join(picture_dir, timestamped_filename)

    # 保存图片
    fig.savefig(filepath, dpi=150)

    print(f"图片已保存到picture文件夹: {filepath}")
    return filepath


def copy_to_osu_songs_dir(audio_path, osu_path, audio_basename):
    """
    将生成的谱面复制到osu!歌曲目录

    参数:
        audio_path: 音频文件路径
        osu_path: .osu文件路径
        audio_basename: 音频文件基本名

    返回:
        目标文件夹路径
    """
    osu_songs_dir = r"D:\osu!\Songs"

    if not os.path.exists(osu_songs_dir):
        print(f"警告: osu!歌曲目录不存在: {osu_songs_dir}")
        return None

    # 创建目标文件夹
    target_folder_name = f"{audio_basename}_automake"
    target_folder = os.path.join(osu_songs_dir, target_folder_name)
    os.makedirs(target_folder, exist_ok=True)

    # 复制音频文件
    audio_filename = os.path.basename(audio_path)
    target_audio_path = os.path.join(target_folder, audio_filename)

    try:
        shutil.copy2(audio_path, target_audio_path)
        print(f"✓ 音频文件复制到: {target_audio_path}")
    except Exception as e:
        print(f"✗ 音频文件复制失败: {e}")

    # 复制.osu文件
    target_osu_path = os.path.join(target_folder, os.path.basename(osu_path))

    try:
        shutil.copy2(osu_path, target_osu_path)
        print(f"✓ 谱面文件复制到: {target_osu_path}")
    except Exception as e:
        print(f"✗ 谱面文件复制失败: {e}")

    print(f"✓ 谱面已复制到osu!歌曲目录: {target_folder}")
    return target_folder


def cleanup_temp_files(temp_dir="temp", max_age_days=7):
    """
    清理超过指定天数的临时文件

    参数:
        temp_dir: 临时目录
        max_age_days: 最大保留天数
    """
    if not os.path.exists(temp_dir):
        return

    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)

    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)

        # 获取文件修改时间
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))

        if mtime < cutoff_time:
            try:
                os.remove(filepath)
                print(f"清理旧文件: {filename}")
            except Exception as e:
                print(f"清理文件失败 {filename}: {e}")
