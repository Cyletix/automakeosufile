import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TEST_SONG_DIR = r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose"
DEFAULT_TEST_AUDIO_FILE = os.path.join(DEFAULT_TEST_SONG_DIR, "Scattered Rose.mp3")
DEFAULT_TEST_REFERENCE_OSU_FILE = os.path.join(
    DEFAULT_TEST_SONG_DIR, "Scattered Rose.osu"
)
DEFAULT_EXPORT_SUBDIR = "automakeosu_generated"


@dataclass
class Config:
    """Runtime configuration for beatmap generation and optimization."""

    SAMPLE_RATE: int = 22050
    DURATION: Optional[float] = None
    MONO: bool = True

    N_FFT: int = 2048
    HOP_LENGTH: int = 512

    N_MELS: int = 128
    FMIN: int = 20
    FMAX: int = 8000

    ADAPTIVE_THRESHOLD_BLOCK_SIZE: int = 15
    ADAPTIVE_THRESHOLD_C: float = -8.0
    MORPH_KERNEL_SIZE: int = 2

    MIN_NOTE_DURATION_MS: int = 5
    MAX_NOTE_DURATION_MS: int = 2000
    NOTE_GAP_MS: int = 10

    BEAT_DIVISORS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16])
    MAX_ALIGN_ERROR_MS: int = 100

    MIN_COLUMN_GAP_MS: int = 1
    MAX_SAME_COLUMN_INTERVAL_MS: int = 10
    PHYSICAL_CORRECTION_STRICTNESS: float = 0.05

    DEFAULT_COLUMNS: int = 7
    COLUMN_MAPPING: Dict[int, List[int]] = field(
        default_factory=lambda: {
            4: [0, 3, 7, 10],
            6: [0, 2, 4, 5, 7, 9],
            7: list(range(7)),
            8: list(range(8)),
        }
    )

    MAX_NOTES_PER_BEAT: int = 16
    ENERGY_SMOOTHING_WINDOW: int = 20
    DENSITY_FILTER_RATIO: float = 0.5
    DENSITY_NPS_SCALE: float = 1.4
    DENSITY_MAPPING: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 20.0),
            (0.2, 22.0),
            (0.4, 24.0),
            (0.6, 26.0),
            (0.8, 28.0),
        ]
    )
    SNAP_RESTRICTIONS: List[Tuple[float, List[int]]] = field(
        default_factory=lambda: [
            (0.0, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.2, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.4, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.6, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.8, [1, 2, 3, 4, 6, 8, 12, 16]),
        ]
    )

    HOLD_NOTE_MIN_DURATION: int = 180
    HOLD_NOTE_MAX_DURATION: int = 800
    HOLD_NOTE_TARGET_PERCENTAGE: float = 15.0

    COLUMN_BALANCE_TARGET_STD: float = 2.0
    COLUMN_REBALANCE_THRESHOLD: float = 0.08

    ENABLE_TIMING_GRID_FILTER: bool = True
    ENABLE_COLUMN_BALANCE_FILTER: bool = True
    ENABLE_SILENCE_ENERGY_FILTER: bool = True

    SILENCE_WINDOW_MS: int = 500
    SILENCE_LEADING_MARGIN_MS: int = 80
    SILENCE_ONSET_ABS_THRESHOLD: float = 0.10
    SILENCE_ONSET_REL_THRESHOLD: float = 1.30
    SILENCE_ABS_THRESHOLD: float = 0.055
    SILENCE_REL_THRESHOLD: float = 0.92

    TIMING_FILTER_HOLD_MIN_DIVISOR: int = 8
    COLUMN_BALANCE_WINDOW_MS: int = 2000
    COLUMN_BALANCE_MAX_SHARE: float = 0.45

    OUTPUT_DIR: Optional[str] = None
    EXPORT_SUBDIR: str = DEFAULT_EXPORT_SUBDIR
    COPY_AUDIO_TO_OUTPUT_DIR: bool = True

    def clone(self, **overrides: Any) -> "Config":
        data = self.to_dict()
        data.update(overrides)
        return Config(**data)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config field: {key}")
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
