"""Metric implemetations and utils."""


def instantiate_metrics(metric_configs, num_classes):
    """Instantiate metric classes with parameters.
    As num_classes is only available in runtime, we can't
    initiate the metric classes directly in the configuration.

    Args:
        metric_configs (list): List of metric configurations.
        num_classes (int): Number of classes for the task.

    Returns:
        list: List of metric instances.
    """
    metric_instances = []
    for metric_config in metric_configs:
        metric_class = metric_config["class"]
        metric_params = metric_config.get("params", {})
        # add num_classes to metric params
        metric_params["num_classes"] = num_classes
        metric_instances.append(metric_class(**metric_params))
    return metric_instances
