__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import metrics
from pim_utils import pim_metrics
from typing import Dict, Any, Callable
import argparse


def net_train(log: dict[str, Any],
              net: nn.Module,
              dataloader: DataLoader,
              optimizer: Optimizer,
              criterion: Callable,
              grad_clip_val: float,
              device: torch.device):
    # Set Network to Training Mode
    net = net.train()
    # Statistics
    losses = []
    # Iterate through batches
    for features, targets in tqdm(dataloader):
        # Move features and targets to the proper device
        features = features.to(device)
        targets = targets.to(device)

        # Initialize all gradients to zero
        optimizer.zero_grad()
        # Forward Propagation
        out = net(features)
        # Calculate the Loss Function
        loss = criterion(out, targets)
        # print('out: ', out)
        # print('loss: ', loss)
        # Backward propagation
        loss.backward()
        # Gradient clipping
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        # Update parameters
        optimizer.step()
        # Detach loss from the graph indicating the end of forward propagation
        loss.detach()
        # Get losses
        losses.append(loss.item())
    # Average loss
    loss = np.mean(losses)
    # Save Statistics
    log['loss'] = loss
    # End of Training Epoch
    return net


def net_eval(log: Dict,
             net: nn.Module,
             dataloader: DataLoader,
             criterion: Callable,
             device: torch.device):
    net = net.eval()
    with torch.no_grad():
        # Statistics
        losses = []
        prediction = []
        ground_truth = []
        # Batch Iteration
        for features, targets in tqdm(dataloader):
            # Move features and targets to the proper device
            features = features.to(device)
            targets = targets.to(device)

            # print('\nfeatures: ', features)
            # print('targets: ', targets)
            # Forward Propagation
            outputs = net(features)
            # Calculate loss function
            loss = criterion(outputs, targets)
            # Collect prediction and ground truth for metric calculation
            prediction.append(outputs.cpu())
            ground_truth.append(targets.cpu())
            # Collect losses to calculate the average loss per epoch
            losses.append(loss.item())
    # Average loss per epoch
    avg_loss = np.mean(losses)
    # Prediction and Ground Truth
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    # Save Statistics
    log['loss'] = avg_loss
    # End of Evaluation Epoch
    return net, prediction, ground_truth

def net_pred(net: nn.Module,
             dataloader: DataLoader,
             device: torch.device):
    net = net.eval()
    with torch.no_grad():
        prediction = []
        # Batch Iteration
        for features, targets in tqdm(dataloader):
            # Move features and targets to the proper device
            features = features.to(device)
            # Forward Propagation
            outputs = net(features)
            prediction.append(outputs.cpu())
    # Average loss per epoch
    # Prediction and Ground Truth
    prediction = torch.cat(prediction, dim=0).numpy()
    # End of Evaluation Epoch
    return prediction
    
def calculate_metrics(args: argparse.Namespace, stat: Dict[str, Any], prediction: np.ndarray, ground_truth: np.ndarray,
                     noise: Dict[str, Any], filter: np.ndarray, means, sd):
        
    pred = prediction.copy()
    gt = ground_truth.copy()
    
    pred[..., 0] = (prediction[..., 0].reshape(1, -1)[0]*sd['Y'][0] + means['Y'][0])
    pred[..., 1] = (prediction[..., 1].reshape(1, -1)[0]*sd['Y'][1] + means['Y'][1])
    gt[..., 0] = (ground_truth[..., 0].reshape(1, -1)[0]*sd['Y'][0] + means['Y'][0])
    gt[..., 1] = (ground_truth[..., 1].reshape(1, -1)[0]*sd['Y'][1] + means['Y'][1])
    
    stat['NMSE'] = metrics.NMSE(prediction, ground_truth)
    # stat['Main_metrics'] = pim_metrics.main_metrics(pred, gt, FS = args.FS, FC_TX = args.FC_TX, PIM_SFT = args.PIM_SFT, PIM_total_BW = args.PIM_total_BW)
    
    stat['Reduction_level'] = pim_metrics.reduction_level(pred, gt, FS = args.FS, FC_TX = args.FC_TX, PIM_SFT = args.PIM_SFT, PIM_BW = args.PIM_BW, noise = noise, filter = filter)
    return stat
