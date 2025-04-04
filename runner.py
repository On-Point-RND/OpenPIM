import os
import random as rnd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils import util
from modules.loggers import PandasLogger


import pyrallis
from config import Config

from pim_utils.data_utils import prepare_residuals, load_for_pred, load_resources
from modules.paths import gen_dir_paths, gen_file_paths
from modules.train_funcs import train_model


class Runner:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Dictionary for Statistics Log

        # Load Hyperparameters
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
        print("::: Best Model Save Path: ", self.args.path_save_file_best)
        print("::: Log-History     Path: ", self.args.path_log_file_hist)
        print("::: Log-Best        Path: ", self.args.path_log_file_best)

        # Instantiate Logger for Recording Training Statistics
        self.logger = PandasLogger(
            path_save_file_best=self.args.path_save_file_best,
            path_log_file_best=self.args.path_log_file_best,
            path_log_file_hist=self.args.path_log_file_hist,
            precision=self.args.log_precision,
        )

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
        print(
            "::: Are Deterministic Algorithms Enabled: ",
            torch.are_deterministic_algorithms_enabled(),
        )
        print("--------------------------------------------------------------------")

    def set_device(self):
        # Find Available GPUs
        if self.args.accelerator == "cuda" and torch.cuda.is_available():
            idx_gpu = self.args.devices
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = torch.device("cuda:" + str(idx_gpu))
            torch.cuda.set_device(device)
            print("::: Available GPUs: %s" % (torch.cuda.device_count()))
            print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            print(
                "--------------------------------------------------------------------"
            )
        elif self.args.accelerator == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.args.accelerator == "cpu":
            device = torch.device("cpu")
            print("::: Available GPUs: None")
            print(
                "--------------------------------------------------------------------"
            )
        else:
            raise ValueError(
                f"The select device {self.args.accelerator} is not supported."
            )
        self.device = device
        return device

    def prepare_residuals(self, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        prepare_residuals(
            data,
            self.args.n_back,
            self.args.n_fwd,
            self.args.n_iterations,
            self.args.batch_size,
            self.args.batch_size_eval,
            train_ratio,
            val_ratio,
            test_ratio,
        )

    def load_for_pred(self):
        return load_for_pred(
            self.args.dataset_path,
            self.args.dataset_name,
            self.args.path_dir_save,
            self.args.n_back,
            self.args.n_fwd,
            self.args.batch_size_eval,
        )

    def load_resources(self):
        return load_resources(
            self.args.dataset_path,
            self.args.dataset_name,
            self.args.filter_path,
            self.args.train_ratio,
            self.args.val_ratio,
            self.args.test_ratio,
            self.args.n_back,
            self.args.n_fwd,
            self.args.batch_size,
            self.args.batch_size_eval,
            path_dir_save=self.path_dir_log_best,
        )

    def build_model(self):
        # Load Pretrained Model if Running Retrain
        if self.args.step == "retrain":
            net = self.net_retrain.Model(self)  # Instantiate Retrain Model
            if self.path_net_pretrain is None:
                print("::: Loading pretrained model: ", self.default_path_net_pretrain)
                # net = util.load_model(self, net, self.default_path_net_pretrain)
                net.load_pretrain_model(self.default_path_net_pretrain)
            else:
                print("::: Loading pretrained model: ", self.path_net_pretrain)
                net = util.load_model(self, net, self.path_net_pretrain)
        else:
            net = self.net_pretrain.Model(self)  # Instantiate Pretrain Model

        # Cast net to the target device
        net.to(self.device)
        self.add_arg("net", net)

        return net

    def build_criterion(self):
        dict_loss = {"l2": nn.MSELoss(), "l1": nn.L1Loss()}
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
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.args.decay_factor,
            patience=self.args.patience,
            verbose=True,
            threshold=1e-4,
            min_lr=self.args.lr_end,
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
        pim_model_id,
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
            device=self.args.devices,
            path_dir_save=self.path_dir_save,
            path_dir_log_hist=self.path_dir_log_hist,
            path_dir_log_best=self.path_dir_log_best,
            pim_model_id=pim_model_id,
            FS=spec_dictionary["FS"],
            FC_TX=spec_dictionary["FC_TX"],
            PIM_SFT=spec_dictionary["PIM_SFT"],
            PIM_BW=spec_dictionary["PIM_BW"],
            n_log_steps=self.args.n_log_steps,
            n_iterations=self.args.n_iterations,
            grad_clip_val=self.args.grad_clip_val,
            lr_schedule=self.args.lr_schedule,
            log_precision=self.args.log_precision,
            save_results=self.args.save_results,
            val_ratio=self.args.log_precision,
            test_ratio=self.args.log_precision,
            PIM_backbone=self.args.log_precision,
            PIM_hidden_size=self.args.log_precision,
            n_back=self.args.log_precision,
            n_fwd=self.args.log_precision,
            batch_size=self.args.log_precision,
        )
