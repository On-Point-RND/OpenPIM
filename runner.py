import os
import random as rnd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from modules.loggers import PandasLogger


import pyrallis
from config import Config

from modules.data_collector import load_resources, load_and_split_data
from modules.paths import gen_dir_paths, gen_file_paths
from modules.train_funcs import train_model
from modules.loggers import make_logger
from modules.loss import IQComponentWiseLoss, HybridLoss, JointLoss, FFTLoss

from modules.data_cascaded import prepare_dataloaders
from torch.optim.lr_scheduler import LinearLR, SequentialLR


class Runner:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Dictionary for Statistics Log

        # Load Hyperparameters
        self.step_logger = make_logger()
        self.args = pyrallis.parse(config_class=Config)
        # Hardware Info
        self.num_cpu_threads = os.cpu_count()

        # Configure Reproducibility
        self.reproducible()

        dir_paths = gen_dir_paths(self.args)

        self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best = dir_paths
        [os.makedirs(p, exist_ok=True) for p in dir_paths]

    def gen_model_id(self, n_net_params):
        dict_pa = {
            "B": f"{self.args.n_back}",
            "F": f"{self.args.n_fwd}",
            "S": f"{self.args.seed}",
            "M": self.args.PIM_backbone.upper(),
            "H": f"{self.args.PIM_hidden_size:d}",
            "P": f"{n_net_params:d}",
        }
        dict_pamodel_id = dict(list(dict_pa.items()))

        list_pamodel_id = []
        for item in list(dict_pamodel_id.items()):
            list_pamodel_id += list(item)
        pa_model_id = "_".join(list_pamodel_id)
        pa_model_id = "PIM_" + pa_model_id
        return pa_model_id

    def build_logger(self, model_id: str):
        # Get Save and Log Paths
        file_paths = gen_file_paths(
            self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best, model_id
        )
        (
            self.args.path_save_file_best,
            self.args.path_log_file_hist,
            self.args.path_log_file_best,
        ) = file_paths
        self.step_logger.info(
            f"::: Best Model Save Path:  {self.args.path_save_file_best}"
        )
        self.step_logger.info(
            f"::: Log-History     Path: {self.args.path_log_file_hist}"
        )
        self.step_logger.info(
            f"::: Log-Best        Path: {self.args.path_log_file_best}"
        )

        # Instantiate Logger for Recording Training Statistics
        PandasWriter = PandasLogger(
            path_save_file_best=self.args.path_save_file_best,
            path_log_file_best=self.args.path_log_file_best,
            path_log_file_hist=self.args.path_log_file_hist,
            precision=self.args.log_precision,
        )

        return PandasWriter

    def reproducible(self):
        rnd.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        # torch.autograd.set_detect_anomaly(True)

        if self.args.re_level == "soft":
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
        else:  # re_level == 'hard'
            torch.use_deterministic_algorithms(mode=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        self.step_logger.info(
            f"::: Are Deterministic Algorithms Enabled:  {torch.are_deterministic_algorithms_enabled()}"
        )

    def set_device(self):
        # Find Available GPUs
        if self.args.accelerator == "cuda" and torch.cuda.is_available():
            idx_gpu = self.args.devices
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = torch.device("cuda:" + str(idx_gpu))
            torch.cuda.set_device(device)
            self.step_logger.info(
                "::: Available GPUs: %s" % (torch.cuda.device_count())
            )
            self.step_logger.info("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            self.step_logger.info(
                "--------------------------------------------------------------------"
            )
        elif self.args.accelerator == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.args.accelerator == "cpu":
            device = torch.device("cpu")
            self.step_logger.info("::: Available GPUs: None")
            self.step_logger.info(
                "--------------------------------------------------------------------"
            )
        else:
            raise ValueError(
                f"The select device {self.args.accelerator} is not supported."
            )
        self.device = device
        return device

    def load_resources(self):
        return load_resources(
            self.args.dataset_path,
            self.args.dataset_name,
            self.args.filter_path,
            self.args.PIM_type,
            self.args.data_type,
            self.args.train_ratio,
            self.args.val_ratio,
            self.args.test_ratio,
            self.args.n_back,
            self.args.n_fwd,
            self.args.batch_size,
            self.args.batch_size_eval,
            path_dir_save=self.path_dir_log_best,
        )

    def build_criterion(self):
        dict_loss = {
            "joint": JointLoss(),
            "hybrid": HybridLoss(),
            "angle": IQComponentWiseLoss(),
            "l2": nn.MSELoss(reduction="mean"),
            "l1": nn.L1Loss(),
            "fft": FFTLoss(),
        }
        loss_func_name = self.args.loss_type
        try:
            criterion = dict_loss[loss_func_name]
            self.criterion = criterion
            return criterion
        except AttributeError:
            raise AttributeError("Please use a valid loss function. Check argument.py.")

    def build_optimizer(self, net: nn.Module):
        # Optimizer
        if self.args.opt_type == "adam":
            optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.opt_type == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt_type == "rmsprop":
            optimizer = optim.RMSprop(net.parameters(), lr=self.args.lr)
        elif self.args.opt_type == "adamw":
            optimizer = optim.AdamW(net.parameters(), lr=self.args.lr)
        elif self.args.opt_type == "adabound":
            import adabound  # Run pip install adabound (https://github.com/Luolc/AdaBound)

            optimizer = adabound.AdaBound(
                net.parameters(), lr=self.args.lr, final_lr=0.1
            )
        else:
            raise RuntimeError("Please use a valid optimizer.")

        # Learning Rate Scheduler
        if self.args.lr_schedule_type == "rop":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=self.args.decay_factor,
                patience=self.args.patience,
                threshold=1e-4,
                min_lr=self.args.lr_end,
            )
        elif self.args.lr_schedule_type == "cosine":
            # Warmup for first 5% of training
            warmup_steps = int(0.05 * self.args.n_iterations)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.n_log_steps,
                eta_min=self.args.lr * 1e-3,
                last_epoch=-1,
            )

            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            raise ValueError(
                f"Please use a valid learning rate scheduler."
            )
        return optimizer, lr_scheduler

    def train(
        self,
        net,
        criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        test_loader,
        best_model_metric,
        noise,
        filter,
        CScaler,
        n_channel_id,
        spec_dictionary,
        writer,
        data_type,
    ):

        return train_model(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            best_model_metric=best_model_metric,
            noise=noise,
            filter=filter,
            CScaler=CScaler,
            n_channel=n_channel_id,
            device=self.device,
            path_dir_save=self.path_dir_save,
            path_dir_log_hist=self.path_dir_log_hist,
            path_dir_log_best=self.path_dir_log_best,
            writer=writer,
            data_type=data_type,
            data_name=self.args.dataset_name,
            FS=spec_dictionary["FS"],
            FC_TX=spec_dictionary["FC_TX"],
            PIM_SFT=spec_dictionary["PIM_SFT"],
            PIM_BW=spec_dictionary["PIM_BW"],
            n_log_steps=self.args.n_log_steps,
            n_iterations=self.args.n_iterations,
            grad_clip_val=self.args.grad_clip_val,
            lr_schedule=self.args.lr_schedule,
            lr_schedule_type=self.args.lr_schedule_type,
            save_results=self.args.save_results,
            val_ratio=self.args.val_ratio,
            test_ratio=self.args.test_ratio,
        )

    def prepare_dataloaders(self, data):
        return prepare_dataloaders(
            data,
            self.args.n_back,
            self.args.n_fwd,
            self.args.batch_size,
            self.args.batch_size_eval,
            path_dir_save=self.path_dir_log_best,
        )

    def load_and_split_data(self):

        path = os.path.join(
            self.args.dataset_path,
            self.args.dataset_name,
            f"{self.args.dataset_name}.mat",
        )
        return load_and_split_data(
            path,
            self.args.filter_path,
            PIM_type=self.args.PIM_type,
        )
