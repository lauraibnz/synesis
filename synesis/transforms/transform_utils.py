"""Configurable transforms."""


def get_transform(transform_config):
    """Get transform from config."""
    transform_class = transform_config["class"]
    transform_params = transform_config["params"]
    transform = transform_class(**transform_params)
    return transform
