from torch_audiomentations import PitchShift

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
}
