import json
import statistics
import sys
from collections import defaultdict

# Expect file paths as command-line arguments
file_paths = sys.argv[1:]
if not file_paths:
    print("Usage: python get_mean_results.py file1.json file2.json ...")
    sys.exit(1)


def get_base_run(run_name):
    """Remove the version prefix (everything before and including the first underscore)"""
    parts = run_name.split("_", 1)
    return parts[1] if len(parts) > 1 else run_name


# Load all files and group the mse metrics by base run name
grouped_results = defaultdict(list)
for file_path in file_paths:
    with open(file_path, "r") as f:
        data = json.load(f)
    for run_name, metrics in data.items():
        base = get_base_run(run_name)
        grouped_results[base].append(metrics["mse"])

# Compute the mean for each base run and issue warnings if any value differs >10% from the mean
mean_results = {}
for base, mse_values in grouped_results.items():
    mean_mse = statistics.mean(mse_values)
    mean_results[base] = {"mse": mean_mse}
    for value in mse_values:
        if abs(value - mean_mse) / mean_mse > 0.10:
            print(
                f"Warning: For run '{base}', a metric value ({value}) differs more than 10% from the mean ({mean_mse})"
            )

# Optionally, write out the mean results to a new JSON file
output_file = "mean_results.json"
with open(output_file, "w") as f:
    json.dump(mean_results, f, indent=4)

print(f"Mean results written to {output_file}")
