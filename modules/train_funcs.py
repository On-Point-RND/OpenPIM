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
    noise: Dict[str, Any],
    filter,
    CScaler,
    device: torch.device,
    path_dir_save: str,
    path_dir_log_hist: str,
    path_dir_log_best: str,
    writer,
    data_type: str,
    data_name: str,
    FS: float,
    FC_TX: float,
    PIM_SFT: float,
    PIM_BW: float,
    n_log_steps: int,
    n_iterations: int,
    grad_clip_val: float,
    lr_schedule: bool,
    lr_schedule_type: str,
    save_results: bool = True,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple:
    """Standalone training function detached from class"""

    step_logger = make_logger()

    # Create directories if they don't exist
    os.makedirs(path_dir_save, exist_ok=True)
    os.makedirs(path_dir_log_hist, exist_ok=True)
    os.makedirs(path_dir_log_best, exist_ok=True)

    start_time = time.time()
    net.train()
    losses = []

    red_levels = []
    mean_red_levels_for_iter = []

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
        # Check the presence of auxiliary loss
        if net.get_aux_loss_state():
            out, aux_loss = net(features)
        else:
            out = net(features)

        if log_shape:
            log_shape = False
            step_logger.info(f"out shape: {out.shape} target shape: {targets.shape}")
        conv_targets = net.filter(targets)

        loss = criterion(out, conv_targets)
        if net.get_aux_loss_state():
            loss += aux_loss
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
                        noise[phase_name],
                        filter,
                        data_type,
                        data_name,
                        CScaler,
                        FS,
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
                plot_spectrums(
                    toComplex(pred),
                    toComplex(gt),
                    FS,
                    FC_TX,
                    PIM_SFT,
                    PIM_BW,
                    iteration,
                    logs["test"]["Reduction_level"],
                    data_type,
                    path_dir_save,
                    cut=False,
                    phase_name=phase_name,
                )
                plot_final_spectrums(
                    toComplex(pred),  
                    toComplex(gt), 
                    toComplex(noise["test"]),
                    FS,
                    FC_TX,
                    PIM_SFT,
                    PIM_BW,
                    iteration,
                    data_type,
                    path_dir_save,
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
            red_level_for_iter = calculate_mean_red(list(red_levels.values()))
            mean_red_levels_for_iter.append(
                [
                    red_level_for_iter,
                    iteration,
                ]
            )

            # Learning rate & model saving
            if lr_schedule:
                if lr_schedule_type == "cosine":
                    lr_scheduler.step()
                elif lr_schedule_type == "rop":
                    lr_scheduler.step(logs["train"]["loss"])
            if save_results:
                writer.save_best_model(net, log_epoch, logs["test"], "loss")

        log_epoch += 1
        if iteration > n_iterations:
            break

    step_logger.info("Training Completed\n")

    powers = dict()
    for key, value in (("gt", gt), ("err", gt - pred), ("noise", noise["test"])):
        compl = toComplex(value)
        powers[key] = [
            compute_power(compl[:, id], data_type, FS, PIM_SFT, PIM_BW, data_name)
            for id in range(compl.shape[1])
        ]

    mean_red_level = calculate_mean_red(list(red_levels.values()))
    max_red_level = max(red_levels.values())

    pd.DataFrame(
        mean_red_levels_for_iter,
        columns=["Mean_red_levels", "Iteration"],
    ).to_csv(path_dir_save + f"/quality_for_iter__seed_{seed}.csv")

    plot_total_perf(
        powers,
        max_red_level,
        mean_red_level,
        path_dir_save,
    )

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
            if net.get_aux_loss_state():
                outputs, _ = net(features)
            else:
                outputs = net(features)
            # Calculate loss function
            conv_targets = net.filter(targets)
            loss = criterion(outputs, conv_targets)

            # Collect prediction and ground truth for metric calculation
            prediction.append(outputs.cpu())
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
