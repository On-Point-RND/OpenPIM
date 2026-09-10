import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable

from tqdm import tqdm
from modules.metrics import (
    calculate_metrics,
    calculate_mean_red,
    compute_powers_dict,
    compute_powers_dict_lite,
    compute_spectra_bundle,
    perf_from_powers,
    perf_from_powers_lite,
    plot_spectrums,
    plot_final_spectrums,
    plot_total_perf,
    plot_total_perf_lite,
)
from modules.experiment_log import ExperimentRecorder
from modules.loggers import make_logger
from modules.data_utils import convert_to_serializable, toComplex


def prepare_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch (1, T, C, 2) -> features unchanged, targets (T, C, 2)."""
    return features, targets.squeeze(0)

def _current_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0


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
    writer,
    data_type: str,
    data_name: str,
    FS: float,
    FC_TX: float,
    PIM_SFT: float,
    PIM_BW: float,
    n_log_steps: int,
    n_lr_steps: int,
    n_iterations: int,
    grad_clip_val: float,
    lr_scheduler_type: str,
    save_results: bool = True,
    plot_per_step_spectrums: bool = False,
    primary: bool = True,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> None:
    """Standalone training function detached from class.

    primary=True (first seed): checkpoint, barplots, spectra, optional PNG spectra.
    primary=False: quality metrics only (append to run_metrics.csv).
    """

    step_logger = make_logger()
    os.makedirs(path_dir_save, exist_ok=True)

    save_results = primary and save_results
    plot_per_step_spectrums = primary and plot_per_step_spectrums
    append_metrics = not primary

    recorder = ExperimentRecorder(
        path_dir_save, seed=seed, append_metrics=append_metrics
    )
    signal_specs = (FS, PIM_SFT, PIM_BW, data_type, data_name)

    start_time = time.time()
    net.train()
    losses = []

    phases = {"val": val_ratio, "test": test_ratio}
    loaders = {"val": val_loader, "test": test_loader}
    logs = {"val": dict(), "test": dict(), "train": dict()}

    log_shape = True
    powers = None
    powers_lite = None
    last_powers_iteration = None
    pred_rescaled = None
    gt_rescaled = None

    for iteration, (features, targets) in enumerate(train_loader):
        features, targets = prepare_batch(features, targets)
        features, targets = features.to(device), targets.to(device)
        if log_shape:
            step_logger.info(
                f"Trainng sample shapes X: {features.shape} Y: {targets.shape}"
            )

        optimizer.zero_grad()
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

        if iteration % n_lr_steps == 0:
            if lr_scheduler_type == "rop":
                lr_scheduler.step(np.mean(losses))
            else:
                lr_scheduler.step()

        if iteration % n_log_steps == 0 and iteration > 0:
            step_logger.info(f"{iteration} iteration out of {n_iterations} is complete")
            logs["train"]["loss"] = float(np.mean(losses))

            for phase_name in phases:
                if phases[phase_name] <= 0:
                    continue
                _, pred, gt = net_eval(
                    logs[phase_name],
                    net,
                    loaders[phase_name],
                    criterion,
                    device,
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
                mean_reduction = calculate_mean_red(
                    list(logs[phase_name]["Reduction_level"].values())
                )
                step_logger.success(
                    f"Mean Reduction_level {phase_name}: {mean_reduction}"
                )
                step_logger.success(
                    f"Reduction_level {phase_name}: "
                    f"{convert_to_serializable(logs[phase_name]['Reduction_level'])}"
                )

            pred_rescaled = CScaler.rescale(pred, key="Y")
            gt_rescaled = CScaler.rescale(gt, key="Y")
            powers = compute_powers_dict(
                gt_rescaled, pred_rescaled, noise["test"], signal_specs
            )
            powers_lite = compute_powers_dict_lite(
                gt_rescaled, pred_rescaled, signal_specs
            )
            last_powers_iteration = iteration
            perf_list = perf_from_powers(powers)
            mean_reduction = calculate_mean_red(perf_list)
            mean_res_lite = perf_from_powers_lite(powers_lite)

            spectra_kwargs = {}
            if primary:
                freqs, psds = compute_spectra_bundle(
                    gt_rescaled, pred_rescaled, noise["test"], FS
                )
                spectra_kwargs = dict(
                    freqs=freqs,
                    psd_rx=psds["rx"],
                    psd_pred=psds["pred"],
                    psd_err=psds["err"],
                    psd_noise=psds["noise"],
                )

            recorder.log_step(
                iteration=iteration,
                time_min=(time.time() - start_time) / 60,
                lr=_current_lr(optimizer),
                train_loss=logs["train"]["loss"],
                test_loss=float(logs["test"].get("loss", np.nan)),
                nmse_by_ch=logs["test"].get("NMSE", {}),
                reduction_by_ch=logs["test"].get("Reduction_level", {}),
                powers=powers,
                powers_lite=powers_lite,
                mean_reduction=mean_reduction,
                mean_res_lite=mean_res_lite,
                **spectra_kwargs,
            )

            if plot_per_step_spectrums:
                plot_spectrums(
                    toComplex(pred_rescaled),
                    toComplex(gt_rescaled),
                    FS,
                    FC_TX,
                    PIM_SFT,
                    PIM_BW,
                    iteration,
                    logs["test"]["Reduction_level"],
                    path_dir_save,
                    data_type=data_type,
                    phase_name="test",
                )
                plot_final_spectrums(
                    toComplex(pred_rescaled),
                    toComplex(gt_rescaled),
                    toComplex(noise["test"]),
                    FS,
                    FC_TX,
                    PIM_SFT,
                    PIM_BW,
                    iteration,
                    data_type,
                    path_dir_save,
                    phase_name="test",
                )

            if save_results:
                writer.save_best_model(net, logs["test"], "loss")

        if iteration > n_iterations:
            break

    step_logger.info("Training Completed\n")

    if primary:
        recorder.save_spectra(
            meta={"FS": FS, "FC_TX": FC_TX, "PIM_SFT": PIM_SFT, "PIM_BW": PIM_BW}
        )
        step_logger.info(f"Spectra saved to {recorder.spectra_path}")
    step_logger.info(f"Metrics saved to {recorder.metrics_path}")

    if primary:
        if powers is None or powers_lite is None:
            if pred_rescaled is None:
                _, pred, gt = net_eval(
                    logs["test"],
                    net,
                    test_loader,
                    criterion,
                    device,
                )
                pred_rescaled = CScaler.rescale(pred, key="Y")
                gt_rescaled = CScaler.rescale(gt, key="Y")
            powers = compute_powers_dict(
                gt_rescaled, pred_rescaled, noise["test"], signal_specs
            )
            powers_lite = compute_powers_dict_lite(
                gt_rescaled, pred_rescaled, signal_specs
            )
            last_powers_iteration = n_iterations

        recorder.save_barplot_powers(
            powers,
            powers_lite,
            last_powers_iteration if last_powers_iteration is not None else n_iterations,
        )
        step_logger.info(f"Barplot data saved to {recorder.barplot_powers_path}")
        plot_total_perf(powers, path_dir_save)
        plot_total_perf_lite(powers_lite, path_dir_save)


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
        for features, targets in tqdm(dataloader):
            features, targets = prepare_batch(features, targets)
            features = features.to(device)
            targets = targets.to(device)
            if net.get_aux_loss_state():
                outputs, _ = net(features)
            else:
                outputs = net(features)
            conv_targets = net.filter(targets)
            loss = criterion(outputs, conv_targets)

            prediction.append(outputs.cpu())
            ground_truth.append(conv_targets.cpu())
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    log["loss"] = avg_loss
    return net, prediction, ground_truth
