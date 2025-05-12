"""General utility functions."""

import os

import requests
import pandas as pd
from tqdm import tqdm
import json
import wandb


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
    if u is None or u == {}:
        return d
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def download_github_dir(owner, repo, path, save_dir):
    """
    Download a directory from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        path (str): The path to the directory in the repository.
        save_dir (str): The directory to save the downloaded files.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        contents = response.json()

        print(contents)

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        length = len(contents)
        pbar = tqdm(contents, total=length)

        for item in pbar:
            # Update the progress bar description
            pbar.set_description(f"Downloading {item['name']}")
            if item["type"] == "file":
                download_url = item["download_url"]
                file_name = item["name"]
                file_path = os.path.join(save_dir, file_name)
                file_content = requests.get(download_url).content
                with open(file_path, "wb") as file:
                    file.write(file_content)
            elif item["type"] == "dir":
                download_github_dir(
                    owner, repo, item["path"], os.path.join(save_dir, item["name"])
                )
    else:
        print(
            f"Failed to fetch directory contents. Status code: {response.status_code}"
        )


def download_github_file(owner, repo, file_path, save_dir):
    """
    Download a file from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        file_path (str): The path to the file in the repository.
        save_dir (str): The directory to save the downloaded file.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url)

    if response.status_code == 200:
        file_info = response.json()
        if file_info["type"] == "file":
            download_url = file_info["download_url"]
            file_name = os.path.basename(file_path)
            save_path = os.path.join(save_dir, file_name)

            os.makedirs(save_dir, exist_ok=True)

            file_content = requests.get(download_url).content
            with open(save_path, "wb") as file:
                file.write(file_content)
            print(f"Downloaded: {save_path}")
        else:
            print(f"The specified path is not a file: {file_path}")
    else:
        print(f"Failed to fetch file. Status code: {response.status_code}")


def get_artifact(wandb_path):
    """Get the artifact given a "path" in the form
    entity/project/run_id/run_name."""
    entity, project, run_id, run_name = wandb_path.split("/")
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    artifact_base_name = run.logged_artifacts()[0].name
    artifact = wandb.Api().artifact(f"{entity}/{project}/{artifact_base_name}")
    return artifact


def get_metric_from_wandb(run, metric_name):
    """Get a metric from a run's evaluation metrics."""
    for art in run.logged_artifacts():
        if art.type == "run_table" and "evaluation_metrics" in art.name:
            art_name = art.name
    artifact = wandb.Api().artifact(f"{run.entity}/{run.project}/{art_name}")
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/evaluation_metrics.table.json") as f:
        data = json.load(f)["data"]
    for row in data:
        if row[0] == metric_name:
            return row[1]
    return None


def get_wandb_config():
    """Get WandB configuration from environment variables with defaults."""
    return {
        "entity": os.environ.get("WANDB_ENTITY"),  # None means use wandb default user
        "project": os.environ.get("WANDB_PROJECT", "synesis"),
    }
