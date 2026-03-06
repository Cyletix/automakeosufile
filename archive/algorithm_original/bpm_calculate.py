import librosa


def get_bpm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # print("Tempo 1:", tempo)
    first_beat_time, last_beat_time = librosa.frames_to_time(
        (beats[0], beats[-1]), sr=sr
    )
    # print("Tempo 2:", 60/((last_beat_time-first_beat_time)/(len(beats)-1)))
    tempo2 = 60 / ((last_beat_time - first_beat_time) / (len(beats) - 1))
    bpm = int(tempo2)
    print("bpm", bpm)

    return bpm, first_beat_time, last_beat_time


if __name__ == "__main__":
    filename = r"audio\NIGHTFALL.wav"
    y, sr = librosa.load(filename)
    result = get_bpm(y, sr)
    print(result)
