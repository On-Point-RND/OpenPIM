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
from pim_utils.pim_metrics import plot_spectrum
from typing import Dict, Any, Callable
import pyrallis


from dataclasses import asdict
from config import Config
from modules.data_collector import IQSegmentDataset, prepare_data
from modules.paths import gen_log_stat, gen_dir_paths, gen_file_paths
from modules.train_funcs import net_train, net_eval, calculate_metrics

def get_mean_var(data):
    I_X_mean = np.mean(data['X']['Train'][:, 0])
    Q_X_mean = np.mean(data['X']['Train'][:, 1])

    I_Y_mean = np.mean(data['Y']['Train'][:, 0])
    Q_Y_mean = np.mean(data['Y']['Train'][:, 1])

    means = {'X': [I_X_mean, Q_X_mean], 'Y': [I_Y_mean, Q_Y_mean]}

    I_X_sd = np.var(data['X']['Train'][:, 0]) ** (1/2)
    Q_X_sd = np.var(data['X']['Train'][:, 1]) ** (1/2)

    I_Y_sd = np.var(data['Y']['Train'][:, 0]) ** (1/2)
    Q_Y_sd = np.var(data['Y']['Train'][:, 1]) ** (1/2)
    
    sd = {'X': [I_X_sd, Q_X_sd], 'Y': [I_Y_sd, Q_Y_sd]}
        
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
        dict_pa = {'S': f"{self.args.seed}",
                   'M': self.args.PIM_backbone.upper(),
                   'H': f"{self.args.PIM_hidden_size:d}",
                   'P': f"{n_net_params:d}"
                   }
        dict_pamodel_id = dict(list(dict_pa.items()))

 
        list_pamodel_id = []
        for item in list(dict_pamodel_id.items()):
            list_pamodel_id += list(item)
        pa_model_id = '_'.join(list_pamodel_id)
        pa_model_id = 'PIM_' + pa_model_id
        return pa_model_id


    def build_logger(self, model_id: str):
        # Get Save and Log Paths
        file_paths = gen_file_paths(self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best, model_id)
        self.args.path_save_file_best, self.args.path_log_file_hist, self.args.path_log_file_best = file_paths
        print("::: Best Model Save Path: ", self.args.path_save_file_best)
        print("::: Log-History     Path: ", self.args.path_log_file_hist)
        print("::: Log-Best        Path: ", self.args.path_log_file_best)

        # Instantiate Logger for Recording Training Statistics
        self.logger = PandasLogger(path_save_file_best=self.args.path_save_file_best,
                                   path_log_file_best=self.args.path_log_file_best,
                                   path_log_file_hist=self.args.path_log_file_hist,
                                   precision=self.args.log_precision)

    def reproducible(self):
        rnd.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        # torch.autograd.set_detect_anomaly(True)

        if self.args.re_level == 'soft':
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
        else:  # re_level == 'hard'
            torch.use_deterministic_algorithms(mode=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        print("::: Are Deterministic Algorithms Enabled: ", torch.are_deterministic_algorithms_enabled())
        print("--------------------------------------------------------------------")

   
    def set_device(self):
        # Find Available GPUs
        if self.args.accelerator == 'cuda' and torch.cuda.is_available():
            idx_gpu = self.devices
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = torch.device("cuda:" + str(idx_gpu))
            torch.cuda.set_device(device)
            print("::: Available GPUs: %s" % (torch.cuda.device_count()))
            print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            print("--------------------------------------------------------------------")
        elif self.args.accelerator == 'mps' and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.args.accelerator == 'cpu':
            device = torch.device("cpu")
            print("::: Available GPUs: None")
            print("--------------------------------------------------------------------")
        else:
            raise ValueError(f"The select device {self.args.accelerator} is not supported.")
        self.device = device
        return device

    def get_amplitude(IQ_signal):
        I = IQ_signal[:, 0]
        Q = IQ_signal[:, 1]
        power = I ** 2 + Q ** 2
        amplitude = np.sqrt(power)
        return amplitude

    def set_target_gain(input_IQ, output_IQ):
        """Calculate the total energy of the I-Q signal."""
        amp_in = get_amplitude(input_IQ)
        amp_out = get_amplitude(output_IQ)
        max_in_amp = np.max(amp_in)
        max_out_amp = np.max(amp_out)
        target_gain = np.mean(max_out_amp / max_in_amp)
        return target_gain
        
    def load_resources(self):

        # Load Dataset
        path = os.path.join(self.args.dataset_path,self.args.dataset_name,self.args.dataset_name+'.mat')
        
        data = prepare_data(path, 
                            self.args.filter_path, 
                            self.args.train_ratio, 
                            self.args.val_ratio, 
                            self.args.test_ratio)

        for k in data['specs']:
            setattr(self, k, data['specs'][k])
        
        input_size =  (1 + self.args.n_back + self.args.n_fwd) 

        # Define PyTorch Datasets

        means, sd = get_mean_var(data)

        for data_type in ['Train', 'Val', 'Test']:
            data['X'][data_type][:, 0] = (data['X'][data_type][:, 0] - means['X'][0]) / sd['X'][0]
            data['X'][data_type][:, 1] = (data['X'][data_type][:, 1] - means['X'][1]) / sd['X'][1]
            
            data['Y'][data_type][:, 0] = (data['Y'][data_type][:, 0] - means['Y'][0]) / sd['Y'][0]
            data['Y'][data_type][:, 1] = (data['Y'][data_type][:, 1] - means['Y'][1]) / sd['Y'][1]
            
        
        train_set = IQSegmentDataset(data['X']['Train'], 
                                   data['Y']['Train'], 
                                   n_back=self.args.n_back, 
                                   n_fwd=self.args.n_fwd)
        
        val_set = IQSegmentDataset(data['X']['Val'], 
                                   data['Y']['Val'], 
                                   n_back=self.args.n_back, 
                                   n_fwd=self.args.n_fwd)
        
        test_set = IQSegmentDataset(data['X']['Test'], 
                                    data['Y']['Test'], 
                                    n_back=self.args.n_back, 
                                    n_fwd=self.args.n_fwd)
        
        # Define PyTorch Dataloaders
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size_eval, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size_eval, shuffle=False)

        return (train_loader, val_loader, test_loader), input_size, data['N'], data['filter'], means, sd

    def build_model(self):
        # Load Pretrained Model if Running Retrain
        if self.args.step == 'retrain':
            net = self.net_retrain.Model(self)  # Instantiate Retrain Model
            if self.path_net_pretrain is None:
                print('::: Loading pretrained model: ', self.default_path_net_pretrain)
                # net = util.load_model(self, net, self.default_path_net_pretrain)
                net.load_pretrain_model(self.default_path_net_pretrain)
            else:
                print('::: Loading pretrained model: ', self.path_net_pretrain)
                net = util.load_model(self, net, self.path_net_pretrain)
        else:
            net = self.net_pretrain.Model(self)  # Instantiate Pretrain Model

        # Cast net to the target device
        net.to(self.device)
        self.add_arg("net", net)

        return net

    def build_criterion(self):
        dict_loss = {'l2': nn.MSELoss(),
                     'l1': nn.L1Loss()
                     }
        loss_func_name = self.args.loss_type
        try:
            criterion = dict_loss[loss_func_name]
            self.criterion=  criterion
            return criterion
        except AttributeError:
            raise AttributeError('Please use a valid loss function. Check argument.py.')

    def build_optimizer(self, net: nn.Module):
        # Optimizer
        if self.args.opt_type == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.opt_type == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt_type == 'rmsprop':
            optimizer = optim.RMSprop(net.parameters(), lr=self.args.lr)
        elif self.args.opt_type == 'adamw':
            optimizer = optim.AdamW(net.parameters(), lr=self.args.lr)
        elif self.args.opt_type == 'adabound':
            import adabound  # Run pip install adabound (https://github.com/Luolc/AdaBound)
            optimizer = adabound.AdaBound(net.parameters(), lr=self.args.lr, final_lr=0.1)
        else:
            raise RuntimeError('Please use a valid optimizer.')

        # Learning Rate Scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            factor=self.args.decay_factor,
                                                            patience=self.args.patience,
                                                            verbose=True,
                                                            threshold=1e-4,
                                                            min_lr=self.args.lr_end)
        return optimizer, lr_scheduler

    def train(self, net: nn.Module, criterion: Callable, optimizer: optim.Optimizer, lr_scheduler,
              train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, best_model_metric: str,
              noise: Dict[str, Any], filter: np.ndarray, means, sd, save_results = True) -> None:
        # Timer
        start_time = time.time()
        # Epoch loop
        print("Starting training...")

        for epoch in range(self.args.n_epochs):
            # -----------
            # Train
            # -----------
            net = net_train(log=self.log_train,
                            net=net,
                            optimizer=optimizer,
                            criterion=criterion,
                            dataloader=train_loader,
                            grad_clip_val=self.args.grad_clip_val,
                            device=self.device)
                
            # -----------
            # Validation
            # -----------
            if self.args.eval_val:
                _, prediction, ground_truth = net_eval(log=self.log_val,
                                                       net=net,
                                                       criterion=criterion,
                                                       dataloader=val_loader,
                                                       device=self.device)
                self.log_val = calculate_metrics(self, self.log_val, prediction, ground_truth, noise['Val'], filter, means, sd)

            # -----------
            # Test
            # -----------
            if self.args.eval_test:
                _, prediction, ground_truth = net_eval(log=self.log_test,
                                                       net=net,
                                                       criterion=criterion,
                                                       dataloader=test_loader,
                                                       device=self.device)
                self.log_test = calculate_metrics(self, self.log_test, prediction, ground_truth, noise['Test'], filter, means, sd)
                
            ###########################################################################################################
            # Logging & Saving
            ###########################################################################################################

            # Generate Log Dict
            end_time = time.time()
            elapsed_time_minutes = (end_time - start_time) / 60.0            
            self.log_all = gen_log_stat(self.args, elapsed_time_minutes, net, optimizer, epoch, self.log_train,
                                        self.log_val, self.log_test)

            # Write Log
            self.logger.write_log(self.log_all)

            # Save best model
            # best_net = net.dpd_model if self.step == 'train_dpd' else net

            ###########################################################################################################
            # Learning Rate Schedule
            ###########################################################################################################
            # Schedule at the beginning of retrain
            lr_scheduler_criteria = self.log_val[best_model_metric]
            if self.args.lr_schedule:
                lr_scheduler.step(lr_scheduler_criteria)

        
        if save_results:
            self.logger.save_best_model(net, epoch=epoch, val_stat=self.log_val, metric_name=best_model_metric)
            self.logger.write_log(self.log_all)

        _, prediction, ground_truth = net_eval(log=self.log_test,
                                                       net=net,
                                                       criterion=criterion,
                                                       dataloader=test_loader,
                                                       device=self.device)
            
        test_metrics = calculate_metrics(self, self.log_test, prediction, ground_truth, noise['Test'], filter, means, sd)
        print('Reduction_level on test sample: ', test_metrics['Reduction_level'])

        prediction[..., 0] = (prediction[..., 0].reshape(1, -1)[0]*sd['Y'][0] + means['Y'][0])
        prediction[..., 1] = (prediction[..., 1].reshape(1, -1)[0]*sd['Y'][1] + means['Y'][1])
        ground_truth[..., 0] = (ground_truth[..., 0].reshape(1, -1)[0]*sd['Y'][0] + means['Y'][0])
        ground_truth[..., 1] = (ground_truth[..., 1].reshape(1, -1)[0]*sd['Y'][1] + means['Y'][1])

        filt_signal = prediction[..., 0] + 1j * prediction[..., 1]
        initial_signal = ground_truth[..., 0] + 1j * ground_truth[..., 1]

        # print('\n\n\nfilt_signal: ', filt_signal)
        # print('initial_signal: ', initial_signal)
        plot_spectrum(initial_signal, initial_signal - filt_signal, self.FS, self.FC_TX, self.path_dir_save)
        
        # plot_spectrum(initial_signal, initial_signal, self.FS, self.FC_TX, self.PIM_SFT)
        # plot_spectrum(filt_signal, filt_signal, self.FS, self.FC_TX, self.PIM_SFT)
            
        print("Training Completed...")
        print(" ")

        return self.log_all
