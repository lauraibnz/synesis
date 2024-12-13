from synesis.equivariance.parameters import train


def test_train():
    train(
        feature="VGGishMTAT",
        dataset="TinySOL",
        transform="Gain",
        task="default",
    )
