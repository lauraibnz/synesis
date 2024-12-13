from synesis.equivariance.parameters import evaluate, train


def test_train():
    train(
        feature="VGGishMTAT",
        dataset="TinySOL",
        transform="Gain",
        task="default",
        task_config={"training": {"num_epochs": 2}},
    )


def test_evaluate():
    model = train(
        feature="VGGishMTAT",
        dataset="TinySOL",
        transform="Gain",
        task="default",
        task_config={"training": {"num_epochs": 1}},
    )
    evaluate(
        model=model,
        feature="VGGishMTAT",
        dataset="TinySOL",
        transform="Gain",
        task="default",
        task_config={"training": {"num_epochs": 2}},
    )
