"""General utility functions."""

import os

import requests
import pandas as pd
from tqdm import tqdm
import json
import wandb
import torch
import numpy as np
import mir_eval
from collections import defaultdict
from torchmetrics import Metric
from sklearn.metrics import precision_recall_fscore_support as prfs

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

# Define a class to calculate the notewise metrics for the transcriber probe
# This class should return the note onset F1 score
class NoteMetrics(Metric):
    def __init__(self, hop_secs: float): 
        super().__init__()
        self.add_state("note_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("note_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.hop_secs = hop_secs
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
            Update the metrics with the predictions and target.

            Args:
                preds (torch.Tensor): The predicted piano roll [frames, num_pitches] - already thresholded {0, 1}.
                target (torch.Tensor): The target piano roll [frames, num_pitches] - boolean {0, 1}.
        """
        batch_size = preds.shape[0]
        for batch_idx in range(batch_size):
            pred_piano_roll = preds[batch_idx]
            target_piano_roll = target[batch_idx]

            # chop piano_rolls (added this for test set)
            pred_piano_roll = pred_piano_roll[:target_piano_roll.shape[0], :]
            target_piano_roll = target_piano_roll[:pred_piano_roll.shape[0], :]

            score = self.notemetrics(pred_piano_roll, target_piano_roll)
            if score is None:
                continue
            precision = score["Precision_no_offset"]
            recall = score["Recall_no_offset"]
            self.note_precision += precision
            self.note_recall += recall
            self.total += 1

    def get_intervals(self, piano_roll: torch.Tensor, filter=False):
        """
            Get the intervals of the piano roll.

            Args:
                piano_roll (torch.Tensor | np.ndarray): The piano roll to get the intervals from [frames, num_pitches].
                filter (bool): Whether to filter out short notes.
        """
        # Ensure tensor is on CPU and in integer format
        if isinstance(piano_roll, torch.Tensor):
            piano_roll = piano_roll.cpu().detach().int()
        else:
            piano_roll = torch.tensor(piano_roll, dtype=torch.int)
        
        padded_roll = torch.zeros((piano_roll.shape[0] + 2, 128), dtype=torch.int)
        padded_roll[1:-1, :] = piano_roll # store the piano roll in the padded roll. So frame 1 to 2nd to last frame is the piano roll

        # so, basically, we have an onset if the note was not on the previous frame, and it is on the current frame
        # (silence and no silence == onset and no silence and silence == offset)

        # subtract the previous frame from the current frame
        diff = padded_roll[1:, :] - padded_roll[:-1, :]
        onsets = (diff == 1).nonzero() # shape [num_onsets, 2]
        offsets = (diff == -1).nonzero() # shape [num_offsets, 2]

        notes = onsets[:, 1] #+ 21 # convert the note to the MIDI note number
        intervals = torch.cat([onsets[:, 0].unsqueeze(1), offsets[:, 0].unsqueeze(1)], dim=1) # shape [num_intervals, 2]

        intervals_secs = intervals * self.hop_secs # convert the intervals to seconds

        # We will filter out events (onset - offset) durations less than 116ms
        if filter:
            interval_diff = intervals_secs[:, 1] - intervals_secs[:, 0]
            valid_intervals = interval_diff >= 0.116
            notes = notes[valid_intervals]
            intervals_secs = intervals_secs[valid_intervals]
        return notes.cpu().detach().numpy(), intervals_secs.cpu().detach().numpy()

    def compute(self):
        """
            Compute the notewise metrics.
            Returns:
                note_f1_score (float): The note onset F1 score.
        """
        if self.note_precision == 0 and self.note_recall == 0:
            return np.float32(0.0)

        precision = self.note_precision / self.total if self.total > 0 else np.float32(0.0)
        recall = self.note_recall / self.total if self.total > 0 else np.float32(0.0)
        note_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.float32(0.0)
        return note_f1_score
    
    def notemetrics(self, pred_piano_roll: torch.Tensor, target_piano_roll: torch.Tensor):
        """
            Compute the notewise metrics for the predicted and target piano rolls.

            Args:
                pred_piano_roll (torch.Tensor): The predicted piano roll [frames, num_pitches].
                target_piano_roll (torch.Tensor): The target piano roll [frames, num_pitches].
            
            Returns:
                scores (dict): A dictionary containing the notewise metrics.
        """
        # Get notes and intervals for both target and predicted piano rolls
        # Apply same filtering to both for fair comparison (is this the right thing to do?)
        target_notes, target_intervals = self.get_intervals(target_piano_roll, filter=True)
        pred_notes, pred_intervals = self.get_intervals(pred_piano_roll, filter=True)

        if len(target_notes) == 0:
            return None
        
        # match notes (reference notes to predicted notes)
        scores = mir_eval.transcription.evaluate(
            target_intervals, target_notes, pred_intervals, pred_notes
        )
        return scores
    

class F1Metrics(Metric):
    def __init__(self, frame_rate: float): 
        super().__init__()
        self.add_state("frame_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("frame_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.frame_rate = frame_rate
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
            Update the metrics with the predictions and target.

            Args:
                preds (torch.Tensor): The predicted piano roll [B, frames, num_pitches].
                target (torch.Tensor): The target piano roll [B, frames, num_pitches].
        """
        # Convert to boolean tensors
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).bool()
        else:
            preds = preds.bool()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).bool()
        else:
            target = target.bool()
        
        batch_size = preds.shape[0]
        for batch_idx in range(batch_size):

            pred_piano_roll = preds[batch_idx]
            target_piano_roll = target[batch_idx]

            # chop pred_piano_roll
            pred_piano_roll = pred_piano_roll[:target_piano_roll.shape[0], :]
            target_piano_roll = target_piano_roll[:pred_piano_roll.shape[0], :]

            # TP = torch.logical_and(pred_piano_roll == True, target_piano_roll == True).sum()
            # FP = torch.logical_and(pred_piano_roll == True, target_piano_roll == False).sum()
            # FN = torch.logical_and(pred_piano_roll == False, target_piano_roll == True).sum()

            # p = TP / (TP + FP + np.finfo(float).eps)
            # r = TP / (TP + FN + np.finfo(float).eps)
            scores = multipitch_metrics(pred_piano_roll.cpu().detach().numpy(), \
                                   target_piano_roll.cpu().detach().numpy(), self.frame_rate)
            p = scores["Precision"]
            r = scores["Recall"]

            self.frame_precision += p
            self.frame_recall += r
            self.total += 1

    def compute(self):
        """
            Compute the framewise metrics.
            Returns:
                frame_f1_score (float): The frame F1 score.
        """
        if self.frame_precision == 0 and self.frame_recall == 0:
            return np.float32(0.0)

        precision = self.frame_precision / self.total if self.total > 0 else np.float32(0.0)
        recall = self.frame_recall / self.total if self.total > 0 else np.float32(0.0)
        frame_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.float32(0.0)
        return frame_f1_score
    
    

def multipitch_metrics(ref_roll: np.ndarray, est_roll: np.ndarray, \
                       frame_rate: float, pitch_offset: int = 0) -> dict:
    """
        Calculate multipitch metrics using mir_eval.

        Args:
        ------
            ref_roll (np.ndarray): Reference multipitch roll.
            est_roll (np.ndarray): Estimated multipitch roll.
            frame_rate (float): Frames per second
            pitch_offset (int): Offset for the pitch values, default is 0.

        Returns:
        -------
            scores (dict): Dictionary containing the calculated metrics.
    """
    time_ref = np.arange(ref_roll.shape[0]) / frame_rate
    time_est = np.arange(est_roll.shape[0]) / frame_rate

    ref_freqs = [np.nonzero(ref_roll[t, :])[0] for t in range(ref_roll.shape[0])]
    est_freqs = [np.nonzero(est_roll[t, :])[0] for t in range(est_roll.shape[0])]

    # Convert frequencies to Hz
    ref_freqs = [np.array([mir_eval.util.midi_to_hz(p+pitch_offset) for p in freqs]) for freqs in ref_freqs]
    est_freqs = [np.array([mir_eval.util.midi_to_hz(p+pitch_offset) for p in freqs]) for freqs in est_freqs]

    
    scores = mir_eval.multipitch.evaluate(
        ref_time=time_ref, ref_freqs=ref_freqs, \
        est_time=time_est, est_freqs=est_freqs
    )

    return scores

class AccMetrics(Metric):
    def __init__(self, frame_rate: float): 
        super().__init__()
        self.add_state("accuracy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.frame_rate = frame_rate
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
            Update the metrics with the predictions and target.

            Args:
                preds (torch.Tensor): The predicted piano roll [B, frames, num_pitches].
                target (torch.Tensor): The target piano roll [B, frames, num_pitches].
        """
        # Convert to boolean tensors
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).bool()
        else:
            preds = preds.bool()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).bool()
        else:
            target = target.bool()
        
        batch_size = preds.shape[0]
        for batch_idx in range(batch_size):

            pred_piano_roll = preds[batch_idx]
            target_piano_roll = target[batch_idx]

            # chop pred_piano_roll
            pred_piano_roll = pred_piano_roll[:target_piano_roll.shape[0], :]
            target_piano_roll = target_piano_roll[:pred_piano_roll.shape[0], :]

            scores = multipitch_metrics(pred_piano_roll.cpu().detach().numpy(), \
                                   target_piano_roll.cpu().detach().numpy(), self.frame_rate)
            a = scores["Accuracy"]

            self.accuracy += a
            self.total += 1

    def compute(self):
        """
            Compute the accuracy metric

            Returns:
                accuracy (float): The accuracy score.
        """
        accuracy = self.accuracy / self.total if self.total > 0 else np.float32(0.0)
        return accuracy
