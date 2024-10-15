"""
Description: 根据mp3文件在同目录下生成对应wav格式文件
Author: Cyletix
Date: 2023-03-17 03:29:08
LastEditTime: 2023-03-21 02:19:17
FilePath: \AutoMakeosuFile\mp3_to_wav.py
"""
import os
import librosa
import soundfile


def mp32wav(mp3_file):
    fname, bac = os.path.splitext(mp3_file)
    if bac == ".mp3":
        y, sr = librosa.load(mp3_file)
        wav_file = os.path.splitext(mp3_file)[0] + ".wav"
        soundfile.write(wav_file, y, sr)
        return wav_file
    else:
        print("this file is not mp3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    # filename = 'E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\YELL!.mp3'

    # 转换audio文件夹下所有文件
    audio_directory = "audio"
    for filename in os.listdir(audio_directory):
        file_path = os.path.join(audio_directory, filename)
        if os.path.isfile(file_path):
            mp32wav(file_path)
