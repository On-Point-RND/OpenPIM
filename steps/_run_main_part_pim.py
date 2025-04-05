import os

import torch
import models as model
from runner import Runner
from utils.util import count_net_params
from modules.data_collector import prepare_data
import pandas as pd
from modules.train_funcs import net_pred
from pim_utils.pim_metrics import plot_spectrum
import numpy as np 

import sys
sys.path.append('../..')

def main(epx: Runner):
    epx.set_device()
    
    # Build Dataloaders
    # (train_loader, val_loader, test_loader), input_size, noise, filter, means, sd = epx.load_resources()
    total_pred_data, total_pred_loaders, input_size, n_channels, total_means, total_sd, FC_TX, FS = epx.load_for_pred()

    net = model.CoreModel(n_channels = n_channels,
                          input_size=input_size,
                          hidden_size=epx.args.PIM_hidden_size,
                          num_layers=epx.args.PIM_num_layers,
                          backbone_type=epx.args.PIM_backbone,
                          batch_size=epx.args.batch_size)

    
    n_net_params = count_net_params(net)
    pim_model_id = epx.gen_model_id(n_net_params)

    for ch in range(len(total_pred_data)):
        
        # Create PIM Output Folder
        path_file_pim_in = os.path.join(epx.path_dir_save, 'CH_' + str(ch), pim_model_id + '.csv')
    
        # Load Pretrained PIM Model
        path_pim_model = os.path.join(epx.path_dir_save, 'CH_' + str(ch), pim_model_id + '.pt')
        checkpoint = torch.load(path_pim_model, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint)
        
        # Move the network to the proper device
        net = net.to(epx.device)
        prediction = net_pred(net,
                 total_pred_loaders[ch],
                 epx.device)
    
        prediction[...,0] = prediction[...,0].flatten() * total_sd[ch]["Y"][0] + total_means[ch]["Y"][0]
        prediction[...,1] = prediction[...,1].flatten() * total_sd[ch]["Y"][1] + total_means[ch]["Y"][1]
        
        old_rxa = total_pred_data[ch]['Y']
        # plot_spectrum(prediction[...,0] + 1j*prediction[...,1],
        #               new_rxa[...,0] + 1j*new_rxa[...,1],
        #               FS, FC_TX, 0, 0, ch, epx.path_dir_save
        #               )
    
        new_rxa = old_rxa.copy()
        new_rxa[:, 0] = old_rxa[:, 0] - prediction[:, 0]
        new_rxa[:, 1] = old_rxa[:, 1] - prediction[:, 1]
    
       
        pim_in = pd.DataFrame({'I_txa': total_pred_data[ch]['X'][:, 0], 'Q_txa': total_pred_data[ch]['X'][:, 1], 
                               'I_rxa_new': new_rxa[:, 0], 'Q_rxa_new': new_rxa[:, 1], 
                               'I_rxa_old': old_rxa[:, 0], 'Q_rxa_old': old_rxa[:, 1], 
                               'I_noise': total_pred_data[ch]['noise'][:, 0], 'Q_noise': total_pred_data[ch]['noise'][:, 1]})
      
        
        pim_in.to_csv(path_file_pim_in, index=False)
        print("PIM outputs saved to the ./pim_out folder.")
