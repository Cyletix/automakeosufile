from OsuFile import OsuFile


def test_osu_parser():
    # Initialize parser with test file
    osu_file = OsuFile("YELL!_test.osu")

    # Parse the file
    osu_file.parse()

    # Verify number of columns
    assert osu_file.num_columns == 6, f"Expected 6 columns, got {osu_file.num_columns}"

    # Verify hit objects were parsed
    assert len(osu_file.hit_objects) > 0, "No hit objects were parsed"
    print(f"Parsed {len(osu_file.hit_objects)} hit objects")

    # Test convert_to_pulse for lane 0
    pulse = osu_file.convert_to_pulse(0)
    print(f"Generated pulse sequence with {len(pulse)} samples")

    print("All tests passed!")


if __name__ == "__main__":
    test_osu_parser()
