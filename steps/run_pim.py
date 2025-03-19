import os

import torch
import models as model
from runner import Runner
from utils.util import count_net_params
from modules.data_collector import prepare_data
import pandas as pd

import sys
sys.path.append('../..')

def main(epx: Runner):
    epx.set_device()

    # Load Dataset
    _, _, _, _, X_test, _ = prepare_data(dataset_path = epx.dataset_path, 
                                         dataset_name=epx.dataset_name)

    # Create PIM Output Folder
    log_path = os.path.join(epx.args.log_our_dir, epx.PIM_backbone, 'validation')
    os.makedirs(log_path,exist_ok=True)
    path_file_pim_in = os.path.join(log_path, pim_model_id + '.csv')


    net_pim = model.CoreModel(input_size=8,  #(n_back+n_fwd+1,2)  # (lag, I and Q)
                              hidden_size=epx.args.PIM_hidden_size,
                              num_layers=epx.args.PIM_num_layers,
                              backbone_type=epx.args.PIM_backbone,
                              batch_size=epx.args.batch_size)

    
    
    n_net_pim_params = count_net_params(net_pim)
    print("::: Number of PIM Model Parameters: ", n_net_pim_params)
    pim_model_id = epx.gen_model_id(n_net_pim_params)

    # Load Pretrained PIM Model
    path_pim_model = os.path.join(epx.args.log_our_dir, 'save', epx.dataset_name, epx.PIM_backbone, 'train_pim', pim_model_id + '.pt')
    checkpoint = torch.load(path_pim_model, map_location=torch.device('cpu'))
    net_pim.load_state_dict(checkpoint)
   
    
    # Move the network to the proper device
    net_pim = net_pim.to(epx.device)


# NOTE: I am not sure how validation is done
   
    net_pim = net_pim.eval()
    with torch.no_grad():
        pim_in = torch.Tensor(X_test).unsqueeze(dim=0).to(epx.device)
        pim_out = net_pim(pim_in)
        pim_out = torch.squeeze(pim_out)
        # Move pim_out to CPU
        pim_out = pim_out.cpu()

    ###########################################################################################################
    # Export Pre-distorted PA Inputs using the Test Set Data
    ###########################################################################################################
    pim_in = pd.DataFrame({'I': X_test[:, 0], 'Q': X_test[:, 1], 'I_pim': pim_out[:, 0], 'Q_pim': pim_out[:, 1]})
  
    
    pim_in.to_csv(path_file_pim_in, index=False)
    print("PIM outputs saved to the ./pim_out folder.")
