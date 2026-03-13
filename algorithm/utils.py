import os
import shutil

from .config import (
    DEFAULT_EXPORT_SUBDIR,
    DEFAULT_TEST_AUDIO_FILE,
    DEFAULT_TEST_REFERENCE_OSU_FILE,
    DEFAULT_TEST_SONG_DIR,
)


def get_default_song_paths():
    return {
        "song_dir": DEFAULT_TEST_SONG_DIR,
        "audio_file": DEFAULT_TEST_AUDIO_FILE,
        "reference_osu_file": DEFAULT_TEST_REFERENCE_OSU_FILE,
        "generated_dir": os.path.join(DEFAULT_TEST_SONG_DIR, DEFAULT_EXPORT_SUBDIR),
    }


def resolve_output_dir(audio_path, output_dir=None, export_subdir=DEFAULT_EXPORT_SUBDIR):
    if output_dir:
        resolved_dir = os.path.abspath(output_dir)
    else:
        audio_dir = os.path.dirname(os.path.abspath(audio_path))
        resolved_dir = os.path.join(audio_dir, export_subdir)

    os.makedirs(resolved_dir, exist_ok=True)
    return resolved_dir


def copy_audio_to_output_dir(audio_path, output_dir):
    source_path = os.path.abspath(audio_path)
    target_path = os.path.join(output_dir, os.path.basename(audio_path))

    if os.path.normcase(source_path) == os.path.normcase(os.path.abspath(target_path)):
        return target_path

    shutil.copy2(source_path, target_path)
    return target_path
