import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
from modules.paths import gen_log_stat

from tqdm import tqdm
from utils.metrics import *

from modules.data_utils import toComplex
from modules.loggers import make_logger


def train_model(
    net: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    best_model_metric: str,
    noise: Dict[str, Any],
    filter,
    CScaler,
    n_channel: int,
    device: torch.device,
    path_dir_save: str,
    path_dir_log_hist: str,
    path_dir_log_best: str,
    writer,
    FS: float,
    FC_TX: float,
    PIM_SFT: float,
    PIM_BW: float,
    n_log_steps: int,
    n_iterations: int,
    grad_clip_val: float,
    lr_schedule: bool,
    save_results: bool = True,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple:
    """Standalone training function detached from class"""

    step_logger = make_logger()

    paths = (path_dir_save, path_dir_log_hist, path_dir_log_best)
    path_dir_save = path_dir_save  # + "/CH_" + str(n_channel)
    path_dir_log_hist = path_dir_log_hist  # + "/CH_" + str(n_channel)
    path_dir_log_best = path_dir_log_best  # + "/CH_" + str(n_channel)

    [
        os.makedirs(p, exist_ok=True)
        for p in [
            path_dir_save,
            path_dir_log_hist,
            path_dir_log_best,
        ]
    ]

    start_time = time.time()
    net.train()
    losses = []

    red_levels = []

    phases = {"val": val_ratio, "test": test_ratio}
    loaders = {"val": val_loader, "test": test_loader}
    logs = {"val": dict(), "test": dict(), "train": dict()}

    log_shape = True
    for iteration, (features, targets) in enumerate(train_loader):
        # if device == "cuda":
        features, targets = features.to(device), targets.to(device)
        if log_shape:
            step_logger.info(
                f"Trainng sample shapes X: {features.shape} Y: {targets.shape}"
            )

        optimizer.zero_grad()
        out = net(features)

        if log_shape:
            log_shape = False
            step_logger.info(f"out shape: {out.shape} target shape: {targets.shape}")
        conv_targets = net.filter(targets)

        loss = criterion(out, conv_targets)
        loss.backward()

        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        optimizer.step()

        losses.append(loss.detach().item())

        log_epoch = 0
        if iteration % n_log_steps == 0 and iteration > 0:
            step_logger.info(f"{iteration} iteration out of {n_iterations} is complete")
            logs["train"]["loss"] = np.mean(losses)

            # Validation/Test evaluation
            for phase_name in phases:
                if phases[phase_name] > 0:

                    _, pred, gt = net_eval(
                        logs[phase_name], net, loaders[phase_name], criterion, device
                    )
                    net.train()
                    logs[phase_name] = calculate_metrics(
                        pred,
                        gt,
                        filter,
                        CScaler,
                        FS,
                        FC_TX,
                        PIM_SFT,
                        PIM_BW,
                        logs[phase_name],
                    )
                mean_reduction = sum(
                    logs[phase_name]["Reduction_level"].values()
                ) / len(logs[phase_name]["Reduction_level"])

                step_logger.success(
                    f"Mean Reduction_level {phase_name}: {mean_reduction}"
                )
                step_logger.success(
                    f"Reduction_level {phase_name}: {logs[phase_name]['Reduction_level']}"
                )

            if phase_name in ["test", "train"] and test_ratio > 0:
                pred = CScaler.rescale(pred, key="Y")
                gt = CScaler.rescale(gt, key="Y")

                for FT in [False, True]:
                    plot_spectrums(
                        toComplex(pred),  # .squeeze(-1),
                        toComplex(gt),  # .squeeze(-1),
                        FS,
                        FC_TX,
                        PIM_SFT,
                        PIM_BW,
                        iteration,
                        logs["test"]["Reduction_level"],
                        path_dir_save,
                        cut=FT,
                        phase_name=phase_name,
                    )

            # Logging
            elapsed_time = (time.time() - start_time) / 60

            log_all = gen_log_stat(
                net,
                optimizer,
                elapsed_time,
                iteration,
                logs["train"],
                logs["train"],
                logs["test"],
            )

            writer.write_log(log_all)

            red_levels = log_all["TEST_REDUCTION_LEVEL"]

            # Learning rate & model saving
            if lr_schedule:
                lr_scheduler.step(logs["test"]["loss"])
            if save_results:
                writer.save_best_model(net, log_epoch, logs["test"], "loss")

        log_epoch += 1
        if iteration > n_iterations:
            break

    step_logger.info("Training Completed\n")

    powers = dict()
    for key, value in (("gt", gt), ("err", gt - pred), ("noise", noise["Test"])):
        compl = toComplex(value)
        powers[key] = [
            compute_power(compl[:, id], FS, FC_TX, PIM_SFT, PIM_BW)
            for id in range(compl.shape[1])
        ]
    
    path_dir_save, path_dir_log_hist, path_dir_log_best = paths
    mean_red_level = np.mean([red_levels[id] for id in red_levels.keys()])
    max_red_level = np.max([red_levels[id] for id in red_levels.keys()])

    plot_total_perf(powers, max_red_level, mean_red_level, path_save = path_dir_save )
    
    return log_all


def net_eval(
    log: Dict,
    net: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
):
    net = net.eval()
    with torch.no_grad():
        losses = []
        prediction = []
        ground_truth = []
        # Batch Iteration
        for features, targets in tqdm(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            outputs = net(features)
            # Calculate loss function
            conv_targets = net.filter(targets)
            loss = criterion(outputs, conv_targets)
            # out_batch_size = outputs.shape[0]
            # loss = criterion(outputs, targets[:out_batch_size, ...])

            # Collect prediction and ground truth for metric calculation
            prediction.append(outputs.cpu())
            # ground_truth.append(targets[:out_batch_size, ...].cpu())
            ground_truth.append(conv_targets.cpu())

            # Collect losses to calculate the average loss per epoch
            losses.append(loss.item())
    # Average loss per epoch
    avg_loss = np.mean(losses)
    # Prediction and Ground Truth
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    # Save Statistics
    log["loss"] = avg_loss
    # End of Evaluation Epoch
    return net, prediction, ground_truth


def calculate_metrics(
    prediction, ground_truth, filter, СScaler, FS, FC_TX, PIM_SFT, PIM_BW, stat
):
    if not "NMSE" in stat:
        stat["NMSE"] = dict()

    if not "Reduction_level" in stat:
        stat["Reduction_level"] = dict()

    n_channels = prediction.shape[1]

    pred = СScaler.rescale(prediction, key="Y")
    gt = СScaler.rescale(ground_truth, key="Y")

    for c in range(n_channels):
        stat["NMSE"][f"CH_{c}"] = NMSE(prediction, ground_truth)

        stat["Reduction_level"][f"CH_{c}"] = reduction_level(
            pred[:, c],
            gt[:, c],
            FS=FS,
            FC_TX=FC_TX,
            PIM_SFT=PIM_SFT,
            PIM_BW=PIM_BW,
            filter=filter,
        )
    return stat
