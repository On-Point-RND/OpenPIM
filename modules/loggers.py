import os

import torch
from loguru import logger

from config import Config


def make_logger():
    logger.add(os.path.join(Config.log_out_dir, "log_file.log"))
    return logger


class CheckpointSaver:
    """Saves best model weights by a scalar criterion (lower is better)."""

    def __init__(self, path_save_file_best: str):
        self.path_save_file_best = path_save_file_best
        self.best_val_metric = None

    def save_best_model(self, net, val_stat, metric_name="loss"):
        best_criteria = val_stat[metric_name]
        if self.best_val_metric is None or best_criteria < self.best_val_metric:
            prev = self.best_val_metric
            self.best_val_metric = best_criteria
            torch.save(net.state_dict(), self.path_save_file_best)
            if prev is None:
                print(
                    f">>> saving best model ({best_criteria} {metric_name}) "
                    f"to {self.path_save_file_best}"
                )
            else:
                print(
                    f">>> saving best model ({prev} -> {best_criteria} {metric_name}) "
                    f"to {self.path_save_file_best}"
                )
