from audiomentations import RoomSimulator, TimeStretch
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
    "PitchShift25": {
        "class": PitchShift,
        "params": {
            "min_transpose_semitones": -12,
            "max_transpose_semitones": -6,
            "p": 1,
            "mode": "per_example",
        },
    },
    "PitchShift50": {
        "class": PitchShift,
        "params": {
            "min_transpose_semitones": -6,
            "max_transpose_semitones": 0,
            "p": 1,
            "mode": "per_example",
        },
    },
    "PitchShift75": {
        "class": PitchShift,
        "params": {
            "min_transpose_semitones": 0,
            "max_transpose_semitones": 6,
            "p": 1,
            "mode": "per_example",
        },
    },
    "PitchShift100": {
        "class": PitchShift,
        "params": {
            "min_transpose_semitones": 6,
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
    "AddWhiteNoise25": {
        "class": AddColoredNoise,
        "params": {
            "min_f_decay": 0,
            "max_f_decay": 0,
            "min_snr_in_db": -30,
            "max_snr_in_db": -10,
            "p": 1,
            "mode": "per_example",
        },
    },
    "AddWhiteNoise50": {
        "class": AddColoredNoise,
        "params": {
            "min_f_decay": 0,
            "max_f_decay": 0,
            "min_snr_in_db": -10,
            "max_snr_in_db": 10,
            "p": 1,
            "mode": "per_example",
        },
    },
    "AddWhiteNoise75": {
        "class": AddColoredNoise,
        "params": {
            "min_f_decay": 0,
            "max_f_decay": 0,
            "min_snr_in_db": 10,
            "max_snr_in_db": 30,
            "p": 1,
            "mode": "per_example",
        },
    },
    "AddWhiteNoise100": {
        "class": AddColoredNoise,
        "params": {
            "min_f_decay": 0,
            "max_f_decay": 0,
            "min_snr_in_db": 30,
            "max_snr_in_db": 50,
            "p": 1,
            "mode": "per_example",
        },
    },
    "TimeStretch": {
        "class": TimeStretch,
        "params": {
            "min_rate": 0.5,
            "max_rate": 2,
            "leave_length_unchanged": False,
            "p": 1,
        },
    },
    "TimeStretch25": {
        "class": TimeStretch,
        "params": {
            "min_rate": 0.5,
            "max_rate": 0.875,
            "leave_length_unchanged": False,
            "p": 1,
        },
    },
    "TimeStretch50": {
        "class": TimeStretch,
        "params": {
            "min_rate": 0.875,
            "max_rate": 1.25,
            "leave_length_unchanged": False,
            "p": 1,
        },
    },
    "TimeStretch75": {
        "class": TimeStretch,
        "params": {
            "min_rate": 1.25,
            "max_rate": 1.625,
            "leave_length_unchanged": False,
            "p": 1,
        },
    },
    "TimeStretch100": {
        "class": TimeStretch,
        "params": {
            "min_rate": 1.625,
            "max_rate": 2.0,
            "leave_length_unchanged": False,
            "p": 1,
        },
    },
    "AddReverb": {
        "class": RoomSimulator,
        "params": {
            "min_target_rt60": 0.0,
            "max_target_rt60": 3.0,
            "min_size_x": 5.6,
            "max_size_x": 5.6,
            "min_size_y": 3.9,
            "max_size_y": 3.9,
            "min_size_z": 3.0,
            "max_size_z": 3.0,
            "min_source_x": 3.5,
            "max_source_x": 3.5,
            "min_source_y": 2.7,
            "max_source_y": 2.7,
            "min_source_z": 2.1,
            "max_source_z": 2.1,
            "min_mic_distance": 0.35,
            "max_mic_distance": 0.35,
            "min_mic_azimuth": 0,
            "max_mic_azimuth": 0,
            "min_mic_elevation": 0,
            "max_mic_elevation": 0,
            "use_ray_tracing": True,
            "calculation_mode": "rt60",
            "max_order": 4,
            "leave_length_unchanged": True,
            "p": 1,
        },
    },
    "HueShift": {
        "min": -0.5,
        "max": 0.5,
    },
    "HueShift25": {
        "min": -0.5,
        "max": -0.25,
    },
    "HueShift50": {
        "min": -0.25,
        "max": 0.0,
    },
    "HueShift75": {
        "min": 0.0,
        "max": 0.25,
    },
    "HueShift100": {
        "min": 0.25,
        "max": 0.5,
    },
    "SaturationShift": {
        "min": -2.0,
        "max": 2.0,
    },
    "SaturationShift25": {
        "min": -2.0,
        "max": -1.0,
    },
    "SaturationShift50": {
        "min": -1.0,
        "max": 0.0,
    },
    "SaturationShift75": {
        "min": 0.0,
        "max": 1.0,
    },
    "SaturationShift100": {
        "min": 1.0,
        "max": 2.0,
    },
    "BrightnessShift": {
        "min": -2.0,
        "max": 2.0,
    },
    "BrightnessShift25": {
        "min": -2.0,
        "max": -1.0,
    },
    "BrightnessShift50": {
        "min": -1.0,
        "max": 0.0,
    },
    "BrightnessShift75": {
        "min": 0.0,
        "max": 1.0,
    },
    "BrightnessShift100": {
        "min": 1.0,
        "max": 2.0,
    },
    "JPEGCompression": {
        "min": 0,
        "max": 100,
    },
}
