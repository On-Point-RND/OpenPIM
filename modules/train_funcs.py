import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
from pim_utils.pim_metrics import compute_power
from modules.paths import gen_log_stat

from tqdm import tqdm
from utils import metrics
from pim_utils import pim_metrics
from pim_utils.pim_metrics import plot_spectrums, compute_power

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
    path_dir_save = path_dir_save + "/CH_" + str(n_channel)
    path_dir_log_hist = path_dir_log_hist + "/CH_" + str(n_channel)
    path_dir_log_best = path_dir_log_best + "/CH_" + str(n_channel)

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

    train_loss_values = []
    test_loss_values = []
    red_levels = []
    all_iterations = []

    phases = {"val": val_ratio, "test": test_ratio}
    loaders = {"val": val_loader, "test": test_loader}
    logs = {"val": dict(), "test": dict(), "train": dict()}

    log_shape = True
    for iteration, (features, targets) in enumerate(train_loader):
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

        loss = criterion(out, targets)
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

                    logs[phase_name] = calculate_metrics(
                        pred,
                        gt,
                        noise[phase_name.title()],
                        filter,
                        CScaler,
                        FS,
                        FC_TX,
                        PIM_SFT,
                        PIM_BW,
                        logs[phase_name],
                    )

                if phase_name == "test" and test_ratio > 0:
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
                        )

                    step_logger.success(
                        f"Reduction_level: {logs['test']['Reduction_level']}"
                    )

            # Logging
            elapsed_time = (time.time() - start_time) / 60

            log_all = gen_log_stat(
                net,
                optimizer,
                elapsed_time,
                iteration,
                logs["train"],
                logs["val"],
                logs["test"],
            )

            writer.write_log(log_all)

            train_loss_values.append(log_all["TRAIN_LOSS"])
            test_loss_values.append(log_all["TEST_LOSS"])
            red_levels.append(log_all["TEST_REDUCTION_LEVEL"])
            all_iterations.append(iteration)

            # Learning rate & model saving
            if lr_schedule:
                lr_scheduler.step(logs["val"]["loss"])
            if save_results:
                writer.save_best_model(net, log_epoch, logs["val"], "loss")

        log_epoch += 1
        if iteration > n_iterations:
            break

    step_logger.info("Training Completed\n")

    loss_dict = {
        "Train loss": train_loss_values,
        "Test loss": test_loss_values,
        "Reduction level": red_levels,
    }
    for k in loss_dict.keys():

        fig = plt.figure(figsize=(10, 7))
        plt.plot(all_iterations, loss_dict[k], linewidth=2, color="red")
        plt.xlabel("Iterations", fontsize=16)
        plt.ylabel(k, fontsize=16)
        plt.grid()
        plt.savefig(f"{path_dir_save}/" + k + ".png", bbox_inches="tight")
        plt.close()

    max_metrics = calculate_metrics(
        pred,
        gt + noise["Test"],
        noise["Test"],
        filter,
        CScaler,
        FS,
        FC_TX,
        PIM_SFT,
        PIM_BW,
        logs["Test"],
    )

    powers = dict()
    for key, value in (("gt", gt), ("err", gt - pred), ("noise", noise["Test"])):
        powers["key"] = compute_power(toComplex(value), FS, FC_TX, PIM_SFT, PIM_BW)

    path_dir_save, path_dir_log_hist, path_dir_log_best = paths
    return (
        log_all,
        loss_dict["Reduction level"][-1],
        max_metrics["Reduction_level"],
        powers,
    )


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
            loss = criterion(outputs, targets)
            # Collect prediction and ground truth for metric calculation
            prediction.append(outputs.cpu())
            ground_truth.append(targets.cpu())
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
    prediction, ground_truth, noise, filter, СScaler, FS, FC_TX, PIM_SFT, PIM_BW, stat
):
    if not "NMSE" in stat:
        stat["NMSE"] = dict()

    if not "Reduction_level" in stat:
        stat["Reduction_level"] = dict()

    n_channels = prediction.shape[1]

    pred = СScaler.rescale(prediction, key="Y")
    gt = СScaler.rescale(ground_truth, key="Y")

    for c in range(n_channels):
        stat["NMSE"][f"CH_{c}"] = metrics.NMSE(prediction, ground_truth)

        stat["Reduction_level"][f"CH_{c}"] = pim_metrics.reduction_level(
            pred[:, c],
            gt[:, c],
            FS=FS,
            FC_TX=FC_TX,
            PIM_SFT=PIM_SFT,
            PIM_BW=PIM_BW,
            noise=noise,
            filter=filter,
        )
    return stat
