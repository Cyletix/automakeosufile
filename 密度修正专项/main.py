from OsuFile import OsuFile
from FrequencyAnalyze import FrequencyAnalyzer
from DensityFix import DensityFixer


def main(osu_path, audio_path):
    osu_file = OsuFile(osu_path)
    osu_file.parse()

    analyzer = FrequencyAnalyzer()
    bpm = analyzer.analyze_bpm(audio_path)
    bpm = 175  # 临时使用固定 BPM
    print(f"BPM: {bpm}")

    # 创建按键密度修正器
    fixer = DensityFixer(bpm=bpm, num_columns=osu_file.num_columns)

    # 修改谱面按键
    modified_hit_objects = fixer.modify_hit_objects(osu_file.hit_objects)

    # 保存修改后的谱面
    osu_file.save(modified_hit_objects)


if __name__ == "__main__":
    osu_path = r"C:\Users\Administrator\AppData\Local\osu!\Songs\2163955 Kurokotei  feat eili - Ceremony -lirile-\Kurokotei  feat. eili - Ceremony -lirile- (kojodat) [Lunar Reverie].osu.a8.osu"
    audio_path = r"C:\Users\Administrator\AppData\Local\osu!\Songs\2163955 Kurokotei  feat eili - Ceremony -lirile-\audio.mp3"
    main(osu_path, audio_path)
