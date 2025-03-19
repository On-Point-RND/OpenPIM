import models as model
from runner import Runner
from utils.util import count_net_params
import pandas as pd
import os

def main(epx: Runner, n_back, n_fwd):
   
    epx.set_device()
    epx.args.n_back = n_back
    epx.args.n_fwd = n_fwd
    
    # Build Dataloaders
    (train_loader, val_loader, test_loader), input_size, noise, filter, means, sd = epx.load_resources()
   
    net = model.CoreModel(input_size=input_size,
                          hidden_size=epx.PIM_hidden_size,
                          num_layers=epx.PIM_num_layers,
                          backbone_type=epx.PIM_backbone,
                          batch_size=epx.batch_size)
    n_net_pim_params = count_net_params(net)
    print("::: Number of PIM Model Parameters: ", n_net_pim_params)
    pim_model_id = epx.gen_pim_model_id(n_net_pim_params)

    net = net.to(epx.device)
   
    epx.build_logger(model_id=pim_model_id)

    criterion = epx.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = epx.build_optimizer(net=net)
    
    ###########################################################################################################
    # Training
    ###########################################################################################################

    total_log = epx.train(net=net,
               criterion=criterion,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               # best_model_metric='NMSE'
               best_model_metric='Reduction_level',
               noise = noise, 
               filter = filter, 
               means = means, 
               sd = sd
              )

    results_folder = './results/grid_search/' + total_log['BACKBONE'] + '/'
    filename = 'total_log' + '__n_back_' + str(total_log['N_BACK']) + '__n_fwd_' + str(total_log['N_FWD']) + '.csv'
    os.makedirs(results_folder, exist_ok=True)
    pd.DataFrame(total_log, index = [0]).to_csv(results_folder + filename, index = False)
