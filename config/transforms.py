from audiomentations import TimeStretch
from torch_audiomentations import AddColoredNoise, PitchShift

configs = {
    "PitchShift": {
        "class": PitchShift,
        "params": {
            "min_transpose_semitones": -12,
            "max_transpose_semitones": 12,
            "p": 1,
            "mode": "per_example",
        },
    },
    "AddWhiteNoise": {
        "class": AddColoredNoise,
        "params": {
            "min_f_decay": 0,
            "max_f_decay": 0,
            "min_snr_in_db": -30,
            "max_snr_in_db": 50,
            "p": 1,
            "mode": "per_example",
        },
    },
    "TimeStretch": {
        "class": TimeStretch,
        "params": {
            "min_rate": 0.8,
            "max_rate": 1.2,
            "p": 1,
        },
    },
}
