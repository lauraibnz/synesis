import re

import wandb

entity = "cplachouras"
project = "synesis"

# Increase API timeout if needed.
api = wandb.Api(timeout=39)

# Get all LibriSpeech runs that have the "2_" prefix in their name.
wandb_runs = [
    run
    for run in api.runs(f"{entity}/{project}")
    if "LibriSpeech" in run.name and "2_" in run.name
]

# Regular expression for EQUI run names.
# This expects the run name structure: "2_<EVAL_TYPE>_<task>_<...>"
equi_pattern = re.compile(r"^(2_)(EQUI_PARA_)(?P<task>[^_]+(?:_[^_]+)?)(_.+)$")

for run in wandb_runs:
    try:
        # Infer task: if run.config already has "task", use it.
        # Otherwise, determine by checking hidden_units in the model config.
        if "task" in run.config:
            inferred_task = run.config["task"]
        else:
            hidden_units = run.config["task_config"]["model"]["params"]["hidden_units"]
            inferred_task = "regression_linear" if not hidden_units else "regression"

        corrected_name = run.name
        match = None
        if "EQUI_PARA_" in run.name:
            match = equi_pattern.match(run.name)

        # If pattern matched then check the extracted task.
        if match:
            current_task = match.group("task")
            if current_task != inferred_task:
                # Replace only the task portion with the inferred task.
                corrected_name = (
                    f"{match.group(1)}{match.group(2)}{inferred_task}{match.group(4)}"
                )
                print(f"Renaming run {run.name} -> {corrected_name}")
                api.update_run(run.id, {"name": corrected_name})
    except Exception as e:
        print(f"✗ Failed to update run {run.name}: {str(e)}")
