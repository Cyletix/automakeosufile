'''
Description: 计算每一个时间点的鼓点强度
Author: Cyletix
Date: 2023-03-14 17:52:37
LastEditTime: 2023-03-15 01:34:47
FilePath: \AutoMakeosuFile\onset detection function.py
'''
import librosa


def my_one_detect(filename):
    # load audio file

    y, sr = librosa.load(filename)

    # compute onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)#表示每个时间点上的音频信号中的突变程度

    # detect onsets
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    # convert onset frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)



if __name__=='__main__':
    filename = 'D:\OneDrive\Code\GitHub\AutoMakeosuFile\YELL!.wav'