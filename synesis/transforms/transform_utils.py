"""Configurable transforms."""


def get_transform(transform_config, **kwargs):
    """Get transform from config."""
    transform_class = transform_config["class"]
    transform_params = transform_config["params"]
    if kwargs:
        transform_params.update(kwargs)
    if transform_class.__name__ != "TimeStretch":
        transform_params["output_type"] = "tensor"
    transform = transform_class(**transform_params)
    return transform
