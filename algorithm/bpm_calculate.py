'''
Description: bpm计算
Author: Cyletix
Date: 2023-03-16 19:15:49
LastEditTime: 2023-08-04 18:16:56
FilePath: \AutoMakeosuFile\bpm_calculate.py
'''
import librosa

def get_bpm(y,sr):
    tempo, beats = librosa.beat.beat_track(y=y,sr=sr)
    # print("Tempo 1:", tempo)  
    first_beat_time, last_beat_time = librosa.frames_to_time((beats[0],beats[-1]),sr=sr)
    # print("Tempo 2:", 60/((last_beat_time-first_beat_time)/(len(beats)-1)))
    tempo2=60/((last_beat_time-first_beat_time)/(len(beats)-1))
    bpm = int(tempo2)

    return bpm,first_beat_time,last_beat_time