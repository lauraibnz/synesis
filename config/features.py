feature_configs = {
    "VGGishMTAT": {
        "__cls__": "VGGishMTAT",
        "item_len_sec": 3.69,
        "sample_rate": 16000,
        "feature_dim": 512,
    },
    'MULE_1728' : {
        "__cls__": "MULE",
        "item_len_sec": 2.99,
        "sample_rate": 16000,
        "feature_dim": 1728,
        "extract_kws": {
            "key": "encoded",
    },
    'MULE_512' : {
        "__cls__": "MULE",
        "item_len_sec": 2.99,
        "sample_rate": 16000,
        "feature_dim": 512,
        "extract_kws": {
            "key": "projected",
        },
    },
}
}