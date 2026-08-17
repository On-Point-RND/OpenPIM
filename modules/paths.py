import os
import argparse


def gen_dir_paths(args: argparse.Namespace):
    path_dir_save = os.path.join(
        args.log_out_dir,
        args.PIM_backbone,
        args.dataset_name,
        args.exp_name,
    )
    return path_dir_save


def gen_file_paths(path_dir_save: str, model_id: str):
    path_file_save = os.path.join(path_dir_save, model_id + ".pt")
    return path_file_save
