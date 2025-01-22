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
            "min_rate": 0.5,
            "max_rate": 2,
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
    "SaturationShift": {
        "min": -2.0,
        "max": 2.0,
    },
    "BrightnessShift": {
        "min": -2.0,
        "max": 2.0,
    },
    "JPEGCompression": {
        "min": 50,
        "max": 100,
    },
}
