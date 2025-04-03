import os
import random as rnd
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable
from torch import optim
from torch.utils.data import DataLoader
from utils import util
from modules.loggers import PandasLogger
from pim_utils.pim_metrics import plot_spectrum, compute_power, reduction_level
from typing import Dict, Any, Callable
import pyrallis


from dataclasses import asdict
from config import Config
from modules.data_collector import InfiniteIQSegmentDataset, IQSegmentDataset, prepare_data, prepare_data_for_predict
from modules.paths import gen_log_stat, gen_dir_paths, gen_file_paths
from modules.train_funcs import net_train, net_eval, calculate_metrics
import matplotlib.pyplot as plt
from scipy.signal import convolve
import pandas as pd

def get_mean_var(data):
    
    means_X = []
    means_Y = []
    sd_X = []
    sd_Y = []
    
    for id in range(len(data["X"]["Train"])):
        I_X_mean = np.mean(data["X"]["Train"][id][:, 0])
        Q_X_mean = np.mean(data["X"]["Train"][id][:, 1])
    
        I_Y_mean = np.mean(data["Y"]["Train"][id][:, 0])
        Q_Y_mean = np.mean(data["Y"]["Train"][id][:, 1])
        
        means_X.append([I_X_mean, Q_X_mean])
        means_Y.append([I_Y_mean, Q_Y_mean])
        
        I_X_sd = np.var(data["X"]["Train"][id][:, 0]) ** (1 / 2)
        Q_X_sd = np.var(data["X"]["Train"][id][:, 1]) ** (1 / 2)
    
        I_Y_sd = np.var(data["Y"]["Train"][id][:, 0]) ** (1 / 2)
        Q_Y_sd = np.var(data["Y"]["Train"][id][:, 1]) ** (1 / 2)

        sd_X.append([I_X_sd, Q_X_sd])
        sd_Y.append([I_Y_sd, Q_Y_sd])

    means = {"X": means_X, "Y": means_Y}
    sd = {"X": sd_X, "Y": sd_Y}

    return means, sd


class Runner:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Dictionary for Statistics Log
        self.log_all = {}
        self.log_train = {}
        self.log_val = {}
        self.log_test = {}

        # Load Hyperparameters
        self.args = pyrallis.parse(config_class=Config)

        # d = asdict(pyrallis.parse(config_class=Config))
        # for k in asdict(pyrallis.parse(config_class=Config)):
        #     print(k, self.args[k])
        #     setattr(self, k, self.args[k])

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
            idx_gpu = self.devices
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

    def get_amplitude(IQ_signal):
        I = IQ_signal[:, 0]
        Q = IQ_signal[:, 1]
        power = I**2 + Q**2
        amplitude = np.sqrt(power)
        return amplitude
        
    def prepare_residuals(self, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):

        txa = np.row_stack((data['I_txa'], data['Q_txa'])).T
        rxa = np.row_stack((data['I_rxa_new'], data['Q_rxa_new'])).T
        initial_rxa = np.row_stack((data['I_rxa_old'], data['Q_rxa_old'])).T
        nfa = np.row_stack((data['I_noise'], data['Q_noise'])).T

        total_samples = txa.shape[0]
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        data = {'X': {'Train': txa[:train_end, :], 'Val': txa[train_end:val_end, :], 'Test': txa[val_end:, :]},
         'Y': {'Train': rxa[:train_end, :], 'Val': rxa[train_end:val_end, :], 'Test': rxa[val_end:, :]},
         'Y_initial': {'Train': initial_rxa[:train_end, :], 'Val': initial_rxa[train_end:val_end, :], 'Test': initial_rxa[val_end:, :]},       
         'N': {'Train': nfa[:train_end, :], 'Val': nfa[train_end:val_end, :], 'Test': nfa[val_end:, :]},}

        input_size = 1 + self.args.n_back + self.args.n_fwd

        # Define PyTorch Datasets

        means_X = [np.mean(data["X"]["Train"][:, 0]), np.mean(data["X"]["Train"][:, 1])]
        means_Y = [np.mean(data["Y"]["Train"][:, 0]), np.mean(data["Y"]["Train"][:, 1])]
        
        sd_X = [np.var(data["X"]["Train"][:, 0]) ** (1 / 2), np.var(data["X"]["Train"][:, 1]) ** (1 / 2)]
        sd_Y = [np.var(data["Y"]["Train"][:, 0]) ** (1 / 2), np.var(data["Y"]["Train"][:, 1]) ** (1 / 2)]

        means = {"X": means_X, "Y": means_Y}
        sd = {"X": sd_X, "Y": sd_Y}

        for data_type in ["Train", "Val", "Test"]:
            data["X"][data_type][:, 0] = (
                data["X"][data_type][:, 0] - means["X"][0]
            ) / sd["X"][0]
            data["X"][data_type][:, 1] = (
                data["X"][data_type][:, 1] - means["X"][1]
            ) / sd["X"][1]

            data["Y"][data_type][:, 0] = (
                data["Y"][data_type][:, 0] - means["Y"][0]
            ) / sd["Y"][0]
            data["Y"][data_type][:, 1] = (
                data["Y"][data_type][:, 1] - means["Y"][1]
            ) / sd["Y"][1]

            data["N"][data_type] = data["N"][data_type][self.args.n_back:-self.args.n_fwd, :]
            data["Y_initial"][data_type] = data["Y_initial"][data_type][self.args.n_back:-self.args.n_fwd, :]

        train_set = InfiniteIQSegmentDataset(
            [data["X"]["Train"]],
            data["Y"]["Train"],
            n_back=self.args.n_back,
            n_fwd=self.args.n_fwd,
        )

        val_set = IQSegmentDataset(
            [data["X"]["Val"]],
            data["Y"]["Val"],
            n_back=self.args.n_back,
            n_fwd=self.args.n_fwd,
        )

        test_set = IQSegmentDataset(
            [data["X"]["Test"]],
            data["Y"]["Test"],
            n_back=self.args.n_back,
            n_fwd=self.args.n_fwd,
        )

        # Define PyTorch Dataloaders
        train_loader = DataLoader(
            train_set, batch_size=self.args.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_set, batch_size=self.args.batch_size_eval, shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=self.args.batch_size_eval, shuffle=False
        )

        return (train_loader, val_loader, test_loader), data["N"], data["Y_initial"], means, sd

    def load_for_pred(self):
        
        path = os.path.join(
            self.args.dataset_path,
            self.args.dataset_name,
            self.args.dataset_name + ".mat",
        )

        data = prepare_data_for_predict(path)
        input_size = 1 + self.args.n_back + self.args.n_fwd
        n_channels = len(data['X'])

        total_means = []
        total_sd = []
        total_pred_data = []
        total_pred_loaders = []
        
        for ch in range(n_channels):
            
            mean_sds = pd.read_csv(self.path_dir_save + '/CH_' + str(ch) + '/means_sd.csv')
            means = {'X': mean_sds['mean_X'].tolist(), 'Y': mean_sds['mean_Y'].tolist()}
            sd = {'X': mean_sds['sd_X'].tolist(), 'Y': mean_sds['sd_Y'].tolist()}

            pred_X = data['X'][ch][self.args.n_back: - self.args.n_fwd,].copy()
            pred_Y = data['Y'][ch][self.args.n_back: - self.args.n_fwd,].copy()
            pred_noise = data['noise'][ch][self.args.n_back: - self.args.n_fwd,].copy()
            pred_data = {'X': pred_X, 'Y': pred_Y, 'noise': pred_noise}
                    
            data['X'][ch][:, 0] = (data['X'][ch][:, 0] - means["X"][0]) / sd["X"][0]
            data['X'][ch][:, 1] = (data['X'][ch][:, 1] - means["X"][1]) / sd["X"][1]
    
            data['Y'][ch][:, 0] = (data['Y'][ch][:, 0] - means["Y"][0]) / sd["Y"][0]
            data['Y'][ch][:, 1] = (data['Y'][ch][:, 1] - means["Y"][1]) / sd["Y"][1]

            pred_set = IQSegmentDataset(
                data['X'],
                data['Y'][ch],
                n_back=self.args.n_back,
                n_fwd=self.args.n_fwd,
            )
    
            pred_loader = DataLoader(
                pred_set, batch_size=self.args.batch_size_eval, shuffle=False
            )

            total_means.append(means)
            total_sd.append(sd)
            total_pred_data.append(pred_data)
            total_pred_loaders.append(pred_loader)
        
        return total_pred_data, total_pred_loaders, input_size, n_channels, total_means, total_sd, data['FC_TX'], data['FS']

    def load_resources(self):

        # Load Dataset
        path = os.path.join(
            self.args.dataset_path,
            self.args.dataset_name,
            self.args.dataset_name + ".mat",
        )

        data = prepare_data(
            path,
            self.args.filter_path,
            self.args.train_ratio,
            self.args.val_ratio,
            self.args.test_ratio,
        )

        for k in data["specs"]:
            setattr(self, k, data["specs"][k])

        input_size = 1 + self.args.n_back + self.args.n_fwd

        # Define PyTorch Datasets

        means, sd = get_mean_var(data)
        n_channels = len(data['X']['Train'])
        
        for data_type in ["Train", "Val", "Test"]:
            for id in range(n_channels):
                data["X"][data_type][id][:, 0] = (
                    data["X"][data_type][id][:, 0] - means["X"][id][0]
                ) / sd["X"][id][0]
                data["X"][data_type][id][:, 1] = (
                    data["X"][data_type][id][:, 1] - means["X"][id][1]
                ) / sd["X"][id][1]
    
                data["Y"][data_type][id][:, 0] = (
                    data["Y"][data_type][id][:, 0] - means["Y"][id][0]
                ) / sd["Y"][id][0]
                data["Y"][data_type][id][:, 1] = (
                    data["Y"][data_type][id][:, 1] - means["Y"][id][1]
                ) / sd["Y"][id][1]

                data["N"][data_type][id] = data["N"][data_type][id][self.args.n_back:-self.args.n_fwd, :]

        all_train_loaders = []
        all_val_loaders = []
        all_test_loaders = []
        
        for id in range(n_channels):

            train_set = InfiniteIQSegmentDataset(
                data["X"]["Train"],
                data["Y"]["Train"][id],
                n_back=self.args.n_back,
                n_fwd=self.args.n_fwd,
                n_iterations=int(self.args.n_iterations)
            )
                
            val_set = IQSegmentDataset(
                data["X"]["Val"],
                data["Y"]["Val"][id],
                n_back=self.args.n_back,
                n_fwd=self.args.n_fwd,
            )
                
            test_set = IQSegmentDataset(
                data["X"]["Test"],
                data["Y"]["Test"][id],
                n_back=self.args.n_back,
                n_fwd=self.args.n_fwd,
            )
                
            # Define PyTorch Dataloaders
            train_loader = DataLoader(
                train_set, batch_size=self.args.batch_size, shuffle=False
            )
            val_loader = DataLoader(
                val_set, batch_size=self.args.batch_size_eval, shuffle=False
            )
            test_loader = DataLoader(
                test_set, batch_size=self.args.batch_size_eval, shuffle=False
            )
            
            all_train_loaders.append(train_loader)
            all_val_loaders.append(val_loader)
            all_test_loaders.append(test_loader)
            
        return (
            (all_train_loaders, all_val_loaders, all_test_loaders),
            input_size,
            n_channels,
            data["N"],
            data["filter"],
            means,
            sd,
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

    def train(self, net: nn.Module, criterion: Callable, optimizer: optim.Optimizer,
            lr_scheduler, train_loader: DataLoader, val_loader: DataLoader,
            test_loader: DataLoader, best_model_metric: str, noise: Dict[str, Any],
            filter: np.ndarray, means, sd, n_channel, model_id, initial_rxa = [], save_results=True) -> None:
                
        paths = (self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best)
        self.path_dir_save = self.path_dir_save + '/CH_' + str(n_channel) 
        self.path_dir_log_hist = self.path_dir_log_hist + '/CH_' + str(n_channel) 
        self.path_dir_log_best = self.path_dir_log_best + '/CH_' + str(n_channel) 
        [os.makedirs(p, exist_ok=True) for p in [self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best]]

        self.build_logger(model_id=model_id)
        
        pd.DataFrame({'Value': ['real', 'imag'], 'mean_X': means['X'], 'mean_Y': means['Y'], 'sd_X': sd['X'], 'sd_Y': sd['Y']}).to_csv(self.path_dir_save + '/means_sd.csv', index=False)
        
        start_time = time.time()
        net.train()
        losses = []

        train_loss_values = []
        test_loss_values = []
        red_levels = []
        n_iterations = []
        
        for iteration, (features, targets) in enumerate(train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            optimizer.zero_grad()            
            
            loss = criterion(net(features), targets)
            loss.backward()
            
            if self.args.grad_clip_val != 0:
                nn.utils.clip_grad_norm_(net.parameters(), self.args.grad_clip_val)
            optimizer.step()
            
            losses.append(loss.detach().item())
            
            log_epoch = 0
            if iteration % self.args.n_log_steps == 0 and iteration > 0:
                print(f'{iteration} iteration out of {self.args.n_iterations} is complete')
                self.log_train['loss'] = np.mean(losses)
                
                # Validation/Test evaluation
                for phase in ['val', 'test']:
                    if getattr(self.args, f"{phase}_ratio", 0) > 0:
                        log = getattr(self, f"log_{phase}")
                        _, pred, gt = net_eval(log, net, eval(f"{phase}_loader"),criterion, self.device)
                        setattr(self, f"log_{phase}", calculate_metrics(
                            self, log, pred, gt, noise[phase.title()], filter, means, sd
                        ))

                    if phase == 'test' and self.args.test_ratio > 0:
                        #pred, gt = self.log_test['pred'], self.log_test['gt']
                        for i in range(2):  # Vectorized unnormalization
                            pred[...,i] = pred[...,i].flatten() * sd["Y"][i] + means["Y"][i]
                            gt[...,i] = gt[...,i].flatten() * sd["Y"][i] + means["Y"][i]
                        
                        # plot_spectrum(
                        #     pred[...,0] + 1j*pred[...,1],
                        #     gt[...,0] + 1j*gt[...,1],
                        #     self.FS, self.FC_TX, iteration, self.log_test['Reduction_level'], n_channel, self.path_dir_save
                        # )
                        plot_spectrum(
                            pred[...,0] + 1j*pred[...,1],
                            gt[...,0] + 1j*gt[...,1],
                            self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW, iteration, self.log_test['Reduction_level'], n_channel,
                            self.path_dir_save
                        )
                        plot_spectrum(
                            pred[...,0] + 1j*pred[...,1],
                            gt[...,0] + 1j*gt[...,1],
                            self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW, iteration, self.log_test['Reduction_level'], n_channel,
                            self.path_dir_save, cut = True
                        )
                        print(f"Reduction_level: {self.log_test['Reduction_level']}")

                        if self.args.step == "train_res":

                            initial_gt = initial_rxa['Test']
                            initial_red_level = reduction_level(initial_gt-pred, initial_gt, FS = self.FS, FC_TX = self.FC_TX, 
                                PIM_SFT = self.PIM_SFT, PIM_BW = self.PIM_BW, noise = noise['Test'], filter = filter)
                            
                            plot_spectrum(
                                pred[...,0] + 1j*pred[...,1],
                                gt[...,0] + 1j*gt[...,1],
                                self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW, iteration, initial_red_level, n_channel, 
                                self.path_dir_save, add_name = '_initial', initial = True,
                                initial_ground_truth = initial_gt[...,0] + 1j*initial_gt[...,1]
                            )
                            plot_spectrum(
                                pred[...,0] + 1j*pred[...,1],
                                gt[...,0] + 1j*gt[...,1],
                                self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW, iteration, initial_red_level, n_channel, 
                                self.path_dir_save, add_name = '_initial', cut = True, initial = True,
                                initial_ground_truth = initial_gt[...,0] + 1j*initial_gt[...,1]
                            )

                # Logging                
                elapsed = (time.time()-start_time)/60
                
                self.log_all = gen_log_stat(self.args, elapsed, net, optimizer, iteration, log_epoch,
                                        self.log_train, self.log_val, self.log_test)
                self.logger.write_log(self.log_all)

                train_loss_values.append(self.log_all['TRAIN_LOSS'])
                test_loss_values.append(self.log_all['TEST_LOSS'])
                red_levels.append(self.log_all['TEST_REDUCTION_LEVEL'])
                n_iterations.append(iteration)
                
                # Learning rate & model saving
                if self.args.lr_schedule:
                    lr_scheduler.step(self.log_val[best_model_metric])
                if save_results:
                    self.logger.save_best_model(net, log_epoch, self.log_val, best_model_metric)
                
              
            log_epoch+=1
            if iteration > self.args.n_iterations: break
                
        print("Training Completed\n")

        loss_dict = {'Train loss': train_loss_values, 'Test loss': test_loss_values, 'Reduction level': red_levels}
        for k in loss_dict.keys():
            
            fig = plt.figure(figsize = (10, 7))
            plt.plot(n_iterations, loss_dict[k], linewidth = 2, color = 'red')
            plt.xlabel('Iterations', fontsize = 16)
            plt.ylabel(k, fontsize = 16)
            plt.grid()
            plt.savefig(f'{self.path_dir_save}/' + k + '.png', bbox_inches='tight')
            plt.close()

        max_metrics = calculate_metrics(self, self.log_test, gt + noise['Test'], gt, noise['Test'], filter, means, sd)

        gt_power = compute_power(gt[...,0] + 1j*gt[...,1], self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW)
        err_power = compute_power((gt - pred)[...,0] + 1j*(gt - pred)[...,1], self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW)
        noise_power = compute_power(noise['Test'][...,0] + 1j*noise['Test'][...,1], self.FS, self.FC_TX, self.PIM_SFT, self.PIM_BW)

        powers = {'gt': gt_power, 'err': err_power, 'noise': noise_power}

        self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best = paths
        return self.log_all, loss_dict['Reduction level'][-1], max_metrics['Reduction_level'], powers
