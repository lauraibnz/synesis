from synesis.equivariance.parameters import train


def test_train():
    train(
        feature="VGGishMTAT",
        dataset="TinySOL",
        transform="Gain",
        task="default",
        task_config={"training": {"num_epochs": 2}},
    )
