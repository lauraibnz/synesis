import wandb

entity = "cplachouras"
project = "synesis"

# Get all runs on LibriSpeech from W&B
api = wandb.Api(timeout=39)
wandb_runs = [
    run for run in api.runs(f"{entity}/{project}") if "LibriSpeech" in run.name
]
wandb_runs = [run for run in wandb_runs if "2_" in run.name]

for run in wandb_runs:
    try:
        # Infer task: if run.config already has "task", use it.
        # Otherwise, determine by checking hidden_units in the model config.
        if "task" in run.config:
            inferred_task = run.config["task"]
        else:
            hidden_units = run.config["task_config"]["model"]["params"]["hidden_units"]
            # If hidden_units is empty we assume "regression_linear", otherwise "regression"
            inferred_task = "regression_linear" if not hidden_units else "regression"

        # Check if the run name includes the inferred task.
        parts = run.name.split("_")
        if len(parts) > 1 and inferred_task not in parts:
            # Insert inferred_task as the second field (after the eval type)
            corrected_name = f"{parts[0]}_{inferred_task}_{'_'.join(parts[1:])}"
        else:
            corrected_name = run.name

        if corrected_name != run.name:
            print(f"Renaming run {run.name} -> {corrected_name}")
            # Update the run name on wandb. The API call below uses the run ID.
            api.update_run(run.id, {"name": corrected_name})
        else:
            print(f"Run {run.name} already correctly named")
    except Exception as e:
        print(f"✗ Failed to update run {run.name}: {str(e)}")
