import models as model
from runner import Runner
from utils.util import count_net_params
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def main(epx: Runner):
   
    epx.set_device()
    
    # Build Dataloaders
    (all_train_loaders, all_val_loaders, all_test_loaders), input_size, n_channels, noise, filter, means, sd = epx.load_resources()
       
    net = model.CoreModel(n_channels = n_channels,
                          input_size=input_size,
                          hidden_size=epx.args.PIM_hidden_size,
                          num_layers=epx.args.PIM_num_layers,
                          backbone_type=epx.args.PIM_backbone,
                          batch_size=epx.args.batch_size)
    
    n_net_pim_params = count_net_params(net)
    print("::: Number of PIM Model Parameters: ", n_net_pim_params)
    pim_model_id = epx.gen_model_id(n_net_pim_params)

    net = net.to(epx.device)
   
    # epx.build_logger(model_id=pim_model_id)

    criterion = epx.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = epx.build_optimizer(net=net)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    CH_test_perf = []
    CH_max_perf = []
    total_powers = {'gt': [], 'err': [], 'noise': []}

    for id in range(n_channels):
        train_loader = all_train_loaders[id]
        val_loader = all_val_loaders[id]
        test_loader = all_test_loaders[id]
        log, test_perf, max_perf, powers = epx.train(net=net,
                   criterion=criterion,
                   optimizer=optimizer,
                   lr_scheduler=lr_scheduler,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   test_loader=test_loader,
                   best_model_metric='Reduction_level',
                   noise = {'Train': noise['Train'][id], 'Val': noise['Val'][id], 'Test': noise['Test'][id]}, 
                   filter = filter, 
                   means = {'X': means['X'][id], 'Y': means['Y'][id]}, 
                   sd = {'X': sd['X'][id], 'Y': sd['Y'][id]},
                   n_channel = id,
                   model_id = pim_model_id
                  )
        CH_test_perf.append(test_perf)
        CH_max_perf.append(max_perf)
        total_powers['gt'].append(powers['gt'])
        total_powers['err'].append(powers['err'])
        total_powers['noise'].append(powers['noise'])

    power_df = pd.DataFrame({
    'RXA':total_powers['gt'],
    'ERR':total_powers['err'],
    'NFA':total_powers['noise']
    })
    fig = plt.figure(figsize = (10, 7))
    power_df.plot.bar(color = ('red', 'blue', 'black'))
    plt.title(f'PIM: ORIG: {round( np.mean(total_powers['gt']), 2)}, RES: {round( np.mean([CH_max_perf[i] - CH_test_perf[i] for i in range(len(CH_test_perf))]), 2)}; Performance ABS: {round( np.max(CH_test_perf), 2)}, MEAN: {round( np.mean(CH_test_perf), 2)}')
    plt.xlabel('Channel number', fontsize = 16)
    plt.ylabel('Signal level [dB]', fontsize = 16)
    plt.savefig(f'{epx.path_dir_save}/' 'barplot_perfofmance.png', bbox_inches='tight')
    plt.close()
    
    
            