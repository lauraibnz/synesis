"""Methods for training and evaluating downstream models."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.features import configs as feature_configs
from config.informativeness.downstream import configs as task_configs
from synesis.datasets.dataset_utils import AggregateDataset, SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.metrics import instantiate_metrics
from synesis.probes import get_probe
from synesis.utils import deep_update, get_artifact, get_wandb_config
import numpy as np
import mir_eval

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(
    feature: str,
    dataset: str,
    task: str,
    label: str,
    task_config: Optional[dict] = None,
    item_format: str = "feature",
    device: Optional[str] = None,
    logging: bool = False,
):
    """
    Train a downstream model.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        label: Factor of variation/label to return
        device: Device to use for training (defaults to "cuda" if available).
        logging: Whether to log to wandb.

    Returns:
        If logging is True, returns the wandb run path to the model artifact.
        Otherwise, returns the trained model.
    """
    seed_everything(42)
    feature_config = feature_configs[feature]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )
    # Set up logging
    if logging:
        run_name = f"INFO_DOWN_{task}_{dataset}_{label}_{feature}"
        wandb_config = get_wandb_config()
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=run_name,
            config={
                "feature": feature,
                "feature_config": feature_config,
                "dataset": dataset,
                "task": task,
                "task_config": task_config,
                "item_format": item_format,
            },
        )
        artifact = wandb.Artifact(run_name, type="model", metadata={"task": task})

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        label=label,
        split="train",
        download=False,
        item_format=item_format,
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        label=label,
        split="validation",
        download=False,
        item_format=item_format,
    )

    if train_dataset[0][0].dim() > 2 and dataset != "ImageNet":
        # If item is 3D, this is a dataset that returns items with subitems
        # (e.g. for audio).
        if task_config["training"]["feature_aggregation"]:
            # If feature_aggreation, we'll wrap the dataset so that it returns
            # aggregated features
            aggregated_train = AggregateDataset(
                train_dataset, feature_extractor_name=feature
            )
            aggregated_val = AggregateDataset(
                val_dataset, feature_extractor_name=feature
            )
            del train_dataset, val_dataset
            train_dataset = aggregated_train
            val_dataset = aggregated_val
        else:
            # If not feature_aggregation, we'll wrap the dataset so that it behaves
            # as a subitem dataset
            wrapped_train = SubitemDataset(train_dataset)
            wrapped_val = SubitemDataset(val_dataset)
            del train_dataset, val_dataset
            train_dataset = wrapped_train
            val_dataset = wrapped_val

    dataloader = DataLoader(
        train_dataset,
        batch_size=task_config["training"]["batch_size"],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=task_config["training"]["batch_size"],
        shuffle=False,
    )

    # if raw_data  (e.g. audio) is being returned from dataset,
    # extract features on-the-fly
    # (the AggregateDatset wrapper also computes features)
    if item_format == "raw" and not task_config["training"]["feature_aggregation"]:
        extractor = get_feature_extractor(feature)
        extractor.to(device)

    # train setup
    if task_config["model"]["type"] == "transcriber":
        n_outputs = 128
    else:
        n_outputs = (
            1
            if task_config["model"]["type"] == "regressor"
            else len(train_dataset.label_encoder.classes_)
        )


    sample_item, _ = train_dataset[0]
    if sample_item.dim() == 1:
        in_features = sample_item.shape[0]
    else:
        in_features = sample_item.shape[1]

    if sample_item.dim() == 3:
        use_temporal_pooling = True
    else:
        use_temporal_pooling = False

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=in_features,
        n_outputs=n_outputs,
        use_temporal_pooling=use_temporal_pooling,
        **task_config["model"]["params"],
    ).to(device)
    criterion = task_config["training"]["criterion"]()
    optimizer_class = task_config["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(), **task_config["training"]["optimizer"]["params"]
    )

    val_metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=n_outputs,
    )

    # train and validation loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    num_epochs = task_config["training"]["num_epochs"]
    patience = task_config["training"]["patience"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for item, target in progress_bar:
            item = item.to(device)

            if dataset != "MPE":
                target = target.to(device)
            else:
                target, mask = target
                target = target.to(device)
                mask = mask.to(device)

            if (
                item_format == "raw"
                and task_config["training"]["feature_aggregation"] is False
            ):
                with torch.no_grad():
                    item = extractor(item)
                    # if channels eaten up, unsqueeze
                    if item.dim() == 2:
                        item = item.unsqueeze(1)
                    if item.device != device:
                        item = item.to(device)
            optimizer.zero_grad()
            output = model(item)
            if len(output.shape) == 2 and n_outputs == 1:
                output = output.squeeze(1)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_for_loss = target.float()
            else:
                target_for_loss = target

            if dataset != "MPE":
                loss = criterion(output, target_for_loss)
            else:
                loss = criterion(output, target_for_loss, mask)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)

            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if logging:
                # Log training metrics
                wandb.log({"train/loss": loss.item()})

        model.eval()
        val_loss = 0
        val_metric_results = {}
        with torch.no_grad():
            for item, target in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                item = item.to(device)

                if dataset != "MPE":
                    target = target.to(device)
                else:
                    target, mask = target
                    target = target.to(device)
                    mask = mask.to(device)

                if (
                    item_format == "raw"
                    and not task_config["training"]["feature_aggregation"]
                ):
                    with torch.no_grad():
                        item = extractor(item)
                        # if channels eaten up, unsqueeze
                        if item.dim() == 2:
                            item = item.unsqueeze(1)
                        if item.device != device:
                            item = item.to(device)

                val_output = model(item)
                if len(val_output.shape) == 2:
                    val_output = val_output.squeeze(1)

                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    target_for_loss = target.float()
                else:
                    target_for_loss = target

                if dataset != "MPE":
                    val_loss += criterion(val_output, target_for_loss).item()
                else:
                    val_loss += criterion(val_output, target_for_loss, mask).item()

                if task_config["model"]["type"] == "regressor":
                    target_for_metrics = target_for_loss
                else:
                    target_for_metrics = target.long()

                for metric_cfg, metric in zip(
                    task_config["evaluation"]["metrics"], val_metrics
                ):
                    metric = metric.to(device)
                    if task_config["model"]["type"] == "transcriber":
                        threshold = 0.2
                        metric.update((val_output >= threshold).int(), target_for_metrics)
                    else:
                        metric.update(val_output, target_for_metrics)

        # Calculate metrics
        avg_val_loss = val_loss / len(val_dataloader)
        for metric_cfg, metric in zip(
            task_config["evaluation"]["metrics"], val_metrics
        ):
            print(f"Computing {metric_cfg['name']} metric...")
            val_metric_results[metric_cfg["name"]] = metric.compute().item()
            metric.reset()
        print(
            f"Epoch {epoch+1}/{num_epochs} -",
            f"Avg train loss: {avg_loss:.4f},",
            f"Avg val loss: {avg_val_loss:.4f}",
        )
        for name, value in val_metric_results.items():
            print(f"{name}: {value:.4f}")

        if logging:
            # Log validation metrics
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    **{
                        f"val/{name}": value
                        for name, value in val_metric_results.items()
                    },
                }
            )

        # Check if the validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the best model
    save_path = Path("ckpt") / "INFO" / "DOWN" / task / dataset / f"{feature}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    if logging:
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb_path = wandb.run.path + "/" + artifact.name
        wandb.finish()
        return wandb_path
    return model


def evaluate(
    model: Union[nn.Module, str],
    feature: str,
    dataset: str,
    task: str,
    label: str,
    task_config: Optional[dict] = None,
    item_format: str = "feature",
    device: Optional[str] = None,
    logging: bool = False,
):
    """
    Evaluate a given trained downstream model.

    Args:
        model: Trained downstream model, or wandb artifact path to model.
               If str is provided, the configs saved online are used, and
               the local ones are ignored.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task.
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        label: Factor of variation/label to return
        device: Device to use for evaluation (defaults to "cuda" if available).
        logging: Whether to log to wandb.

    Returns:
        Dictionary of evaluation metrics.
    """

    if isinstance(model, str):
        # Resume wandb run
        entity, project, run_id, model_name = model.split("/")
        if logging:
            wandb.init(project=project, entity=entity, id=run_id, resume="allow")

    feature_config = feature_configs[feature]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)),
        task_config,
    )

    test_dataset = get_dataset(
            name=dataset,
            feature=feature,
            label=label,
            split="test",
            download=False,
            item_format=item_format,
    )

    if test_dataset[0][0].dim() > 2 and dataset != "ImageNet":
        # If item is 3D, this is a dataset that returns items with subitems
        # (e.g. for audio).
        if task_config["evaluation"]["feature_aggregation"]:
            # If feature_aggreation, we'll wrap the dataset so that it returns
            # aggregated features
            aggregated_test = AggregateDataset(
                test_dataset, feature_extractor_name=feature
            )
            del test_dataset
            test_dataset = aggregated_test
        else:
            # If not feature_aggregation, we'll wrap the dataset so that it behaves
            # as a subitem dataset
            wrapped_test = SubitemDataset(test_dataset)
            del test_dataset
            test_dataset = wrapped_test

    dataloader = DataLoader(
        test_dataset,
        batch_size=task_config["evaluation"]["batch_size"],
        shuffle=False,
    )

    if isinstance(model, str):
        # Load model from wandb artifact
        artifact = get_artifact(model)
        artifact_dir = artifact.download()

        if task_config["model"]["type"] == "transcriber":
            n_outputs = 128
        else:
            n_outputs = (
                1
                if task_config["model"]["type"] == "regressor"
                else len(test_dataset.label_encoder.classes_)
            )

    sample_item, _ = test_dataset[0]
    if sample_item.dim() == 1:
        in_features = sample_item.shape[0]
    else:
        in_features = sample_item.shape[1]

    if sample_item.dim() == 3:
        use_temporal_pooling = True
    else:
        use_temporal_pooling = False

    if isinstance(model, str):
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=in_features,
            n_outputs=n_outputs,
            use_temporal_pooling=use_temporal_pooling,
            **task_config["model"]["params"],
        )
        model.load_state_dict(
            torch.load(Path(artifact_dir) / f"{feature}.pt", weights_only=True)
        )
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=n_outputs,
    )

    # if raw_data  (e.g. audio) is being returned from dataset,
    # extract features on-the-fly
    # (the AggregateDatset wrapper also computes features)
    if item_format == "raw" and not task_config["evaluation"]["feature_aggregation"]:
        extractor = get_feature_extractor(feature)
        extractor.to(device)

    model.eval()
    total_loss = 0
    test_metric_results = {}
    criterion = task_config["evaluation"]["criterion"]()

    with torch.no_grad():
        for item, target in tqdm(dataloader, desc="Evaluating"):
            item = item.to(device)

            if dataset != "MPE":
                target = target.to(device)
            else:
                target, mask = target
                target = target.to(device)
                mask = mask.to(device)

            if (
                item_format == "raw"
                and not task_config["evaluation"]["feature_aggregation"]
            ):
                with torch.no_grad():
                    item = extractor(item)
                    # if channels eaten up, unsqueeze
                    if item.dim() == 2:
                        item = item.unsqueeze(1)
                    if item.device != device:
                        item = item.to(device)

            output = model(item)
            if len(output.shape) == 2 and n_outputs == 1:
                output = output.squeeze(1)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_for_loss = target.float()
            else:
                target_for_loss = target

            if dataset != "MPE":
                total_loss += criterion(output, target_for_loss).item()
            else:
                total_loss += criterion(output, target_for_loss, mask).item()

            if task_config["model"]["type"] == "regressor":
                target_for_metrics = target_for_loss
            else:
                target_for_metrics = target.long()

            for metric_cfg, metric in zip(
                task_config["evaluation"]["metrics"], metrics
            ):
                metric = metric.to(device)
                if task_config["model"]["type"] == "transcriber":
                    threshold = 0.2
                    metric.update((output >= threshold).int(), target_for_metrics)
                else:
                    metric.update(output, target_for_metrics)

    avg_loss = total_loss / len(dataloader)
    for metric_cfg, metric in zip(
        task_config["evaluation"]["metrics"], metrics
    ):
        test_metric_results[metric_cfg["name"]] = metric.compute().item()
        metric.reset()
    print(f"Avg test loss: {avg_loss:.4f}")

    for name, value in test_metric_results.items():
        print(f"{name}: {value:.4f}")

    if logging:
        # Log individual test metrics
        wandb.log({
            **{f"test/{name}": value for name, value in test_metric_results.items()},
        })
        
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Test Loss", avg_loss)
        for name, value in test_metric_results.items():
            metrics_table.add_data(name, value)

        # Log the table to wandb
        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return test_metric_results

def stitch(arr: np.ndarray):
    # Stitches the results from the model
    # so that everything aligns
    arr = arr[:, :-1, :]
    result = []
    num_segments, num_frames, _ = arr.shape
    factor_75 = int(num_frames * 0.75)
    factor_25 = int(num_frames * 0.25)
    result.append(arr[0, :factor_75])
    for each in range(1, num_segments - 1):
        result.append(arr[each, factor_25 : factor_75])
    result.append(arr[-1, factor_25:])
    result = np.concatenate(result)
    return result


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

    # filter frequencies above 5000 Hz
    ref_freqs = [freqs[freqs <= 5000] for freqs in ref_freqs]
    est_freqs = [freqs[freqs <= 5000] for freqs in est_freqs]

    # filter frequencies below 20Hz
    ref_freqs = [freqs[freqs >= 20] for freqs in ref_freqs]
    est_freqs = [freqs[freqs >= 20] for freqs in est_freqs]

    
    scores = mir_eval.multipitch.evaluate(
        ref_time=time_ref, ref_freqs=ref_freqs, \
        est_time=time_est, est_freqs=est_freqs
    )

    return scores


def notemetrics(pred_piano_roll: torch.Tensor, target_piano_roll: torch.Tensor, frame_rate: float):
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
        target_notes, target_intervals = get_intervals(target_piano_roll, frame_rate, filter=True)
        pred_notes, pred_intervals = get_intervals(pred_piano_roll, frame_rate, filter=True)

        if len(target_notes) == 0:
            return None
        
        # match notes (reference notes to predicted notes)
        scores = mir_eval.transcription.evaluate(
            target_intervals, target_notes, pred_intervals, pred_notes
        )
        return scores

def get_intervals(piano_roll: torch.Tensor, frame_rate, filter=False):
        """
            Get the intervals of the piano roll.

            Args:
                piano_roll (torch.Tensor | np.ndarray): The piano roll to get the intervals from [frames, num_pitches].
                filter (bool): Whether to filter out short notes.
        """
        # Ensure tensor is on CPU and in integer format
        hop_secs = 1/frame_rate
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

        intervals_secs = intervals * hop_secs # convert the intervals to seconds
        # We will filter out events (onset - offset) durations less than 116ms
        if filter:
            interval_diff = intervals_secs[:, 1] - intervals_secs[:, 0]
            valid_intervals = interval_diff >= 0.116
            notes = notes[valid_intervals]
            intervals_secs = intervals_secs[valid_intervals]
        
        return notes.cpu().detach().numpy(), intervals_secs.cpu().detach().numpy()


def get_test_frame_metrics(model, files, threshold, feature_config):
    frame_metrics = {'p': [], 'r': [], 'a': [], 'f1': []}
    for file in tqdm(files):
        file_npz = np.load(file)
        feature = file_npz["feature"]
        audio_len = file_npz["audio_len"]
        audio_secs = audio_len / feature_config["sample_rate"]
        label = file_npz["label_frames"]
        feature = torch.tensor(feature).to("cuda" if torch.cuda.is_available() else "cpu")
        out_list = []
        with torch.no_grad():
            for feat in feature:
                out = model(feat.unsqueeze(0))
                out_list.append(out.squeeze().cpu().numpy())
        
        output = np.array(out_list)

        # stitch the output
        output = stitch(output)

        # stitch the label too
        label = stitch(label)

        # calculate the frame metrics
        output = output.squeeze() >= threshold
        output = output[:label.shape[0], :]
        label = label[:output.shape[0], :]
        frame_rate = output.shape[0] / audio_secs
        scores = multipitch_metrics(output, label, frame_rate)
        p = scores["Precision"]
        r = scores["Recall"]
        a = scores["Accuracy"]
        f1 = 2 * (p * r) / (p + r + 1e-8)
        frame_metrics['p'].append(p)
        frame_metrics['r'].append(r)
        frame_metrics['a'].append(a)
        frame_metrics['f1'].append(f1)

    p = np.mean(frame_metrics['p'])
    r = np.mean(frame_metrics['r'])
    a = np.mean(frame_metrics['a'])
    f1 = np.mean(frame_metrics['f1'])
    return f1, a

def get_test_note_metrics(model, files, threshold, feature_config):
    note_metrics = {'p_no_offset': [], 'r_no_offset': [], 'f1_no_offset': []}
    for file in files:
        file_npz = np.load(file)
        feature = file_npz["feature"]
        audio_len = file_npz["audio_len"]
        audio_secs = audio_len / feature_config["sample_rate"]
        label = file_npz["label_frames"]
        feature = torch.tensor(feature).to("cuda" if torch.cuda.is_available() else "cpu")
        out_list = []
        with torch.no_grad():
            for feat in feature:
                out = model(feat.unsqueeze(0))
                out_list.append(out.squeeze().cpu().numpy())
        
        output = np.array(out_list)

        # stitch the output
        output = stitch(output)

        # stitch the label too
        label = stitch(label)
        #calculate the note metrics
        frame_rate = output.shape[0] / audio_secs
        output = output.squeeze() >= threshold
        output = output[:label.shape[0], :]
        label = label[:output.shape[0], :]
        note_scores = notemetrics(output, torch.tensor(label), frame_rate)
        if note_scores is not None:
            p_no_offset = note_scores["Precision_no_offset"]
            r_no_offset = note_scores["Recall_no_offset"]
            f1_no_offset = note_scores["F-measure_no_offset"]
            note_metrics['p_no_offset'].append(p_no_offset)
            note_metrics['r_no_offset'].append(r_no_offset)
            note_metrics['f1_no_offset'].append(f1_no_offset)

    p = np.mean(note_metrics['p_no_offset'])
    r = np.mean(note_metrics['r_no_offset'])
    f1 = np.mean(note_metrics['f1_no_offset'])
    return f1

def get_val_metrics(model, files, threshold, feature_config):
    frame_metrics = {'p': [], 'r': [], 'a': [], 'f1': []}
    for file in tqdm(files):
        file_npz = np.load(file)
        feature = file_npz["feature"]
        audio = file_npz["audio"]
        audio_secs = audio.shape[-1] / feature_config["sample_rate"]
        label = file_npz["label_frames"]
        feature = torch.tensor(feature).to("cuda" if torch.cuda.is_available() else "cpu")
        frame_rate = feature.shape[1] / audio_secs
        with torch.no_grad():
            output = model(feature)
        
        # calculate the frame metrics
        output = output.squeeze().cpu().numpy() >= threshold
        output = output[:label.shape[0], :]
        label = label[:output.shape[0], :]
        scores = multipitch_metrics(output, label, frame_rate)
        p = scores["Precision"]
        r = scores["Recall"]
        a = scores["Accuracy"]
        f1 = 2 * (p * r) / (p + r + 1e-8)
        frame_metrics['p'].append(p)
        frame_metrics['r'].append(r)
        frame_metrics['a'].append(a)
        frame_metrics['f1'].append(f1)

    p = np.mean(frame_metrics['p'])
    r = np.mean(frame_metrics['r'])
    a = np.mean(frame_metrics['a'])
    f1 = np.mean(frame_metrics['f1'])
    return f1, a

def evaluate_mpe(
    model: Union[nn.Module, str],
    feature: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    logging: bool = False,
):
    """
    Evaluate a given trained downstream model
    on multi-pitch estimation (MPE) or 
    transcription task.
    Returns:
        Dictionary of evaluation metrics.
    """

    if isinstance(model, str):
        # Resume wandb run
        entity, project, run_id, model_name = model.split("/")
        if logging:
            wandb.init(project=project, entity=entity, id=run_id, resume="allow")

    feature_config = feature_configs[feature]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)),
        task_config,
    )

    if isinstance(model, str):
        # Load model from wandb artifact
        artifact = get_artifact(model)
        artifact_dir = artifact.download()

        if task_config["model"]["type"] == "transcriber":
            n_outputs = 128
        else:
            raise ValueError("MPE evaluation only supports transcription task.")
    
    if feature == "SSVQVAE_Structure":
        in_features = 1024
    elif feature == "SSVQVAE_Combined":
        in_features = 2048
    elif feature == "TSDSAE_Structure":
        in_features = 16
    elif feature == "TSDSAE_Combined":
        in_features = 32
    elif feature == "AFTER_Structure":
        in_features = 12
    elif feature == "AFTER_Combined":
        in_features = 18
    else:
        raise ValueError(f"Feature {feature} not supported for MPE evaluation.")

    if isinstance(model, str):
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=in_features,
            n_outputs=n_outputs,
            **task_config["model"]["params"],
        )
        model.load_state_dict(
            torch.load(Path(artifact_dir) / f"{feature}.pt", weights_only=True)
        )
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    model.eval()
    test_metric_results = {}
    val_data_paths = "./data/MPE/validation"
    val_files = list(Path(val_data_paths).rglob("*.npz"))
    test_data_paths = "./data/MPE/test"
    test_files = list(Path(test_data_paths).rglob("*.npz"))

    # Get the best threshold from validation set
    thresholds = np.arange(0, 1, 0.05)
    selected_threshold = None
    best_f1 = -1
    for thresh in thresholds:
        f1, _ = get_val_metrics(model, val_files, thresh, feature_config)
        if selected_threshold is None or f1 > best_f1:
            best_f1 = f1
            selected_threshold = thresh

    print(f"Selected threshold: {selected_threshold}")

    note_f1 = get_test_note_metrics(model, test_files, 0.1, feature_config) 
    frame_f1, a = get_test_frame_metrics(model, test_files, selected_threshold, feature_config)

    test_metric_results["Note F1"] = note_f1
    test_metric_results["Frame F1"] = frame_f1
    test_metric_results["Frame Accuracy"] = a

    for name, value in test_metric_results.items():
        print(f"{name}: {value:.4f}")

    if logging:
        # Log individual test metrics
        wandb.log({
            **{f"test/{name}": value for name, value in test_metric_results.items()},
        })
        
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        for name, value in test_metric_results.items():
            metrics_table.add_data(name, value)

        # Log the table to wandb
        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return test_metric_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a downstream model.")
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Feature name.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        help="Task name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to use for training.",
    )
    parser.add_argument(
        "--item_format",
        "-i",
        type=str,
        default="feature",
        help="Format of the input data: ['raw', 'feature']. Defaults to 'feature'.",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=str,
        required=True,
        help="Factor of variation or label to predict.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Do not log to wandb.",
    )
    args = parser.parse_args()

    model = train(
        feature=args.feature,
        dataset=args.dataset,
        task=args.task,
        device=args.device,
        label=args.label,
        item_format=args.item_format,
        logging=not args.nolog,
    )

    if args.task == "transcriber_probe":
        results = evaluate_mpe(
            model=model,
            feature=args.feature,
            task=args.task,
            device=args.device,
            logging=not args.nolog,
        )
    else:
        results = evaluate(
            model=model,
            feature=args.feature,
            dataset=args.dataset,
            item_format=args.item_format,
            label=args.label,
            task=args.task,
            device=args.device,
            logging=not args.nolog,
        )
