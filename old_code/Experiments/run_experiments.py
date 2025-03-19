import sys
import os
import pandas as pd
import time
import logging

import numpy as np
from scipy.io import loadmat
from ..Modules.least_squares_models import *
from ..Modules.metrics import *


def run_polynomial_experiment(experiment_mode, experiment_degree, experiment_n_back, experiment_n_fwd, experiment_n_points, 
                              experiment_name, output_dir = '../Results/Polynomial_experiments/',
                              save_results = False, data_path = '5m'):
                            
    print('***************************************************************************************************')
    print(f"Running experiment: {experiment_name}")

    os.makedirs(output_dir, exist_ok=True)
    if data_path == '5m':
        data = loadmat("../Data/FOR_COOPERATION/1TR_C20Nc1CD_E20Ne1CD_20250117_5m/1TR_C20Nc1CD_E20Ne1CD_20250117_5m.mat")
    elif data_path == '0.5m':
        data = loadmat("../Data/FOR_COOPERATION/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m.mat")
    elif data_path == '1L':
        data = loadmat("../Data/FOR_COOPERATION/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L.mat")
    elif data_path == '16L':
        data = loadmat("../Data/FOR_COOPERATION/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L.mat")
    fil = loadmat("../Data/FOR_COOPERATION/rx_filter.mat")

    rxa = data["rxa"]
    txa = data["txa"]
    nfa = data["nfa"]
        
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FC_RX = data['BANDS_UL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6
    PIM_SFT = data['PIM_sft'][0][0] / 10**6
    PIM_BW = data['BANDS_TX'][0][0][1][0][0] / 10**6
    PIM_total_BW = data['BANDS_TX'][0][0][3][0][0] / 10**6

    if experiment_n_points == 0:
        experiment_n_points = txa.shape[1]
    
    train_n_points = int(experiment_n_points*0.8)
    
    logging.basicConfig(filename= output_dir + experiment_name + '_' + data_path + '_experiment.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    start_time = time.time()
    
    win_len = experiment_n_back + experiment_n_fwd + 1
        
    txa_train = txa[:, :train_n_points + win_len -1]
    # rxa_train = rxa[experiment_n_back :experiment_n_back + train_n_points]
    rxa_train = rxa[:, : train_n_points + win_len -1]

    txa_test = txa[:, train_n_points + win_len: ]
    rxa_test = rxa[:, train_n_points + win_len + experiment_n_back : train_n_points + experiment_n_back + txa_test.shape[1] + 1]
    
    pred_pim, w = pim_from_regression(txa_train, rxa_train, fil, mode = experiment_mode, degree=experiment_degree, 
                          n_back = experiment_n_back, n_fwd = experiment_n_fwd, verbose = 0, FS = FS, FC_TX = FC_TX, 
                          PIM_SFT = PIM_SFT, PIM_total_BW = PIM_total_BW, arr_noise = nfa, convolved = False, n_points=train_n_points)

    pim_for_test = predict_pim_with_regression(txa_test, w, fil, mode = experiment_mode, degree = experiment_degree, 
                                               n_back = experiment_n_back, n_fwd = experiment_n_fwd, convolved = False)
    filt_signal = rxa_test - pim_for_test
    filt_signal_train = rxa_train[:, experiment_n_back :rxa_train.shape[1] - experiment_n_fwd] - pred_pim

        
    total_train_metric = []
    total_experiment_metric = []
    for id in range(rxa_train.shape[0]):
        train_metric = calculate_metrics(rxa_train[id], filt_signal_train[id], FS, PIM_SFT, PIM_BW)
        experiment_metric = calculate_metrics(rxa_test[id], filt_signal[id], FS, PIM_SFT, PIM_BW)
        total_train_metric.append(train_metric)
        total_experiment_metric.append(experiment_metric)
    
    end_time = time.time()
    experiment_time = end_time - start_time
    
    logging.info(f"Experiment completed in {experiment_time:.2f} seconds")
    logging.info(f"Metric: {experiment_metric:.2f}")
    if save_results:
        for id in range(rxa_train.shape[0]):
            pd.DataFrame({'Filtered_signal': filt_signal[id], 'Rxa': rxa_test[id]}).to_csv(output_dir + experiment_name +'_' + data_path + '_id_'+ str(id) +'_experiment.tsv')
    
    print('Metric on train sample: ', np.mean(total_train_metric))
    print('Metric on test sample: ', np.mean(total_experiment_metric))

    return np.mean(total_train_metric), np.mean(total_experiment_metric)
