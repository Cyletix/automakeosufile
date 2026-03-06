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
    # filename = r'E:\osu!\Songs\DJ Genki VS Camellia feat moimoi - YELL! [6k]\YELL!.mp3'

    # 转换audio文件夹下所有文件
    audio_directory = os.path.join(os.path.dirname(__file__), "..", "audio")
    for filename in os.listdir(audio_directory):
        file_path = os.path.join(audio_directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".mp3"):
            wav_file = os.path.splitext(file_path)[0] + ".wav"
            if not os.path.exists(wav_file):
                mp32wav(file_path)
