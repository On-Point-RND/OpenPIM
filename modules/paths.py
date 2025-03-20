__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import argparse
import os


def gen_log_stat(args: argparse.Namespace, elapsed_time, net, optimizer, epoch, train_stat=None, val_stat=None,
                 test_stat=None):
    # Get Epoch & Batch Size
    n_iterations = args.n_iterations
    batch_size = args.batch_size

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    if args.step == 'train_pim':
        backbone = args.PIM_backbone
        hidden_size = args.PIM_hidden_size

    # Create log dictionary
    log_stat = {'EPOCH': epoch,
                'n_iterations': n_iterations,
                'TIME:': elapsed_time,
                'LR': lr_curr,
                'BATCH_SIZE': batch_size,
                'N_PARAM': n_param,
                'BACKBONE': backbone,
                'HIDDEN_SIZE': hidden_size,
                'N_BACK' : args.n_back,
                'N_FWD' : args.n_fwd,
                }

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f'TRAIN_{k.upper()}': v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f'VAL_{k.upper()}': v for k, v in val_stat.items()}
        log_stat = {**log_stat, **val_stat_log}
    if test_stat is not None:
        test_stat_log = {f'TEST_{k.upper()}': v for k, v in test_stat.items()}
        log_stat = {**log_stat, **test_stat_log}
    return log_stat


def gen_dir_paths(args: argparse.Namespace):
    path_dir_save = os.path.join(args.log_out_dir,'save', args.dataset_name, args.PIM_backbone, args.step)  # Best model save dir
    path_dir_log_hist = os.path.join(args.log_out_dir,'log', args.dataset_name, args.PIM_backbone, args.step, 'history')  # Log dir to save training history
    path_dir_log_best = os.path.join(args.log_out_dir, 'log', args.dataset_name, args.PIM_backbone, args.step, 'best')  # Log dir to save info of the best epoch
    dir_paths = (path_dir_save, path_dir_log_hist, path_dir_log_best)
    return dir_paths


def gen_file_paths(path_dir_save: str, path_dir_log_hist: str, path_dir_log_best: str, model_id: str):
    # File Paths
    path_file_save = os.path.join(path_dir_save, model_id + '.pt')
    path_file_log_hist = os.path.join(path_dir_log_hist, model_id + '.csv')  # .csv path_log_file_hist
    path_file_log_best = os.path.join(path_dir_log_best, model_id + '.csv')  # .csv path_log_file_hist
    file_paths = (path_file_save, path_file_log_hist, path_file_log_best)
    return file_paths


