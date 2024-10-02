"""General utility functions."""


def deep_update(d, u):
    """
    Recursively update a dict with another (that potentially
    doesn't have all keys)

    Args:
        d: The dict to update.
        u: The dict to update from.

    Returns:
        The updated dict.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
