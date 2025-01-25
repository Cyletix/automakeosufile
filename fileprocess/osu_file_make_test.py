from fileprocess.osu_file_make import OSUGenerator


def test_osu_generator():
    generator = OSUGenerator()
    generator.set_audio_file("test.mp3")
    generator.set_metadata("Test Title", "Test Artist", "Test Creator", "Test Version")
    generator.set_difficulty(5, 4, 5, 5)
    generator.generate_from_analysis(120, [1000, 2000, 3000], 4000)
    generator.save("test.osu")

    with open("test.osu", "r", encoding="utf-8") as f:
        content = f.read()
        assert "osu file format v14" in content
        assert "AudioFilename: test.mp3" in content
        assert "Title:Test Title" in content
        assert "Artist:Test Artist" in content
        assert "Creator:Test Creator" in content
        assert "Version:Test Version" in content
        assert "HPDrainRate:5" in content
        assert "CircleSize:4" in content
        assert "OverallDifficulty:5" in content
        assert "ApproachRate:5" in content
        assert "0,500.0,4,2,1,60,1,0" in content
        assert "256,192,1000,1,0,0:0:0:0:" in content
        assert "256,192,2000,1,0,0:0:0:0:" in content
        assert "256,192,3000,1,0,0:0:0:0:" in content

    print("All tests passed!")


if __name__ == "__main__":
    test_osu_generator()
