import os
import argparse


def gen_log_stat(
    net,
    optimizer,
    elapsed_time,
    n_iterations,
    train_stat=None,
    val_stat=None,
    test_stat=None,
):
    # Get Epoch & Batch Size

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group["lr"]

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # Create log dictionary
    log_stat = {
        "iteration": n_iterations,
        "n_iterations": n_iterations,
        "TIME:": elapsed_time,
        "LR": lr_curr,
        "N_PARAM": n_param,
    }

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f"TRAIN_{k.upper()}": v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f"VAL_{k.upper()}": v for k, v in val_stat.items()}
        log_stat = {**log_stat, **val_stat_log}
    if test_stat is not None:
        test_stat_log = {f"TEST_{k.upper()}": v for k, v in test_stat.items()}
        log_stat = {**log_stat, **test_stat_log}
    return log_stat


def gen_dir_paths(args: argparse.Namespace):
    path_dir_save = os.path.join(
        args.log_out_dir,
        args.PIM_backbone,
        args.dataset_name,
        args.exp_name,
        # "save",
    )  # Best model save dir
    path_dir_log_hist = os.path.join(
        args.log_out_dir,
        args.PIM_backbone,
        args.dataset_name,
        args.exp_name,
        # "log",
        # "history",
    )  # Log dir to save training history
    path_dir_log_best = os.path.join(
        args.log_out_dir,
        args.PIM_backbone,
        args.dataset_name,
        args.exp_name,
        "best",
    )  # Log dir to save info of the best epoch
    dir_paths = (path_dir_save, path_dir_log_hist, path_dir_log_best)
    return dir_paths


def gen_file_paths(
    path_dir_save: str, path_dir_log_hist: str, path_dir_log_best: str, model_id: str
):
    # File Paths
    path_file_save = os.path.join(path_dir_save, model_id + ".pt")
    path_file_log_hist = os.path.join(
        path_dir_log_hist, model_id + ".csv"
    )  # .csv path_log_file_hist
    path_file_log_best = os.path.join(
        path_dir_log_best, model_id + ".csv"
    )  # .csv path_log_file_hist
    file_paths = (path_file_save, path_file_log_hist, path_file_log_best)
    return file_paths
