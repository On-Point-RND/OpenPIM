import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from itertools import product
from scipy.signal import convolve
from scipy.io import loadmat
import json

from linalg_aux import *
from metrics import *


class SignalConfig:
    fs: float
    fc_tx: float
    pim_sft: float
    pim_bw: float
    pim_total_bw: float

    def __init__(self, signal_data):
        self.fc_tx = signal_data['BANDS_DL'][0][0][0][0][0] / 10**6
        self.fs = signal_data['Fs'][0][0] / 10**6
        self.pim_sft = signal_data['PIM_sft'][0][0] / 10**6
        self.pim_bw = signal_data['BANDS_TX'][0][0][1][0][0] / 10**6
        self.pim_total_bw = signal_data['BANDS_TX'][0][0][3][0][0] / 10**6


class ModelConfig:
    def __init__(self, filename):
           with open(filename, 'r') as f:
            config = json.load(f)
            self.model = config['model']
            self.poly = config['poly']
            self.pim_type = config['pim_type']
            self.n_back = config['n_back']
            self.n_fwd = config['n_fwd']


def multrans_model(rxa, txa, conv4metrics, bf_len: int,
                   config: ModelConfig, sig_config: SignalConfig):
    fs, pim_sft, pim_bw = sig_config.fs, sig_config.pim_sft, sig_config.pim_bw
    model_func = globals()[config.model]
    poly_func = globals()[config.poly]
    n_back = config.n_back
    n_fwd = config.n_fwd
    n_cut = int(rxa.shape[0] * 0.8)
    n_trans = rxa.shape[1]
    rxa_train_mem = rxa[:n_cut]
    txa_train_mem = txa[:n_cut]
    rxa_test_mem = rxa[n_cut:]
    txa_test_mem = txa[n_cut:]
    rxa_train = rxa_train_mem[n_back:-n_fwd]
    rxa_test = rxa_test_mem[n_back:-n_fwd]
    n_train = rxa_train.shape[0]
    n_test = rxa_test.shape[0]

    pred_train = np.empty((n_train,n_trans), dtype=np.complex128, order='F')
    pred_test = np.empty((n_test,n_trans), dtype=np.complex128, order='F')

    conv_train = np.empty((n_train+254,n_trans), dtype=np.complex128, order='F')
    conv_test = np.empty((n_test+254,n_trans), dtype=np.complex128, order='F')

    convolve_tensor(rxa_train, conv4metrics, conv_train)
    convolve_tensor(rxa_test, conv4metrics, conv_test)

    conv_pred_train = np.empty((n_train+254,n_trans), dtype=np.complex128, order='F')
    conv_pred_test = np.empty((n_test+254,n_trans), dtype=np.complex128, order='F')

    mtn_train = create_model_tensor(
        model_func, poly_func,
        txa_train_mem, bf_len, n_back, n_fwd
    )

    mtn_test = create_model_tensor(
        model_func, poly_func,
        txa_test_mem,bf_len, n_back, n_fwd
    )
    model_wts = ls_multi_trans(mtn_train, rxa_train)
    pred_train[...] = rxa_train
    pred_test[...] = rxa_test
    contract(mtn_train, model_wts, pred_train)
    contract(mtn_test, model_wts, pred_test)
    convolve_tensor(pred_train, conv4metrics, conv_pred_train)
    convolve_tensor(pred_test, conv4metrics, conv_pred_test)
    train_metric = calculate_avg_metrics(
        conv_train, conv_pred_train, fs, pim_sft, pim_bw
    )
    test_metric = calculate_avg_metrics(
        conv_test, conv_pred_test, fs, pim_sft, pim_bw)
    
    result = dict()
    result["train_metric"] = train_metric
    result["test_metric"] = test_metric
    result["pred_train"] = conv_pred_train
    result["pred_test"] = conv_pred_test
    return result


def singletrans_model():
    return True

def poly_inference(output_dir = './results/',
                              data_marker = '5m'):
    print('*****************************************************************')
    # print(f"Running experiment: {experiment_name}")

    config = ModelConfig('config.json')
    n_back, n_fwd = config.n_back, config.n_fwd
    os.makedirs(output_dir, exist_ok=True)
    multi_trans = False
    if data_marker == '5m':
        data = loadmat("../Data/1TR_C20Nc1CD_E20Ne1CD_20250117_5m.mat")
    elif data_marker == '0.5m':
        data = loadmat("../Data/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m.mat")
    elif data_marker == '1L':
        multi_trans = True
        data = loadmat("../Data/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L.mat")
    elif data_marker == '16L':
        multi_trans = True
        data = loadmat("../Data/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L.mat")

    fil = loadmat("../Data/rx_filter.mat")
    n,m = data["rxa"].shape[1], data["rxa"].shape[0] 
    conv_data = fil['flt_coeff'].flatten()
    rxa = np.empty((n,m), dtype=np.complex128, order='F')
    txa = np.empty((n,m), dtype=np.complex128, order='F')
    if config.pim_type == 'total':
        rxa[...] = np.copy(data["rxa"].T)
    elif config.pim_type == 'ext':
        rxa[...] = np.copy(data["PIM_EXT"].T + data["nfa"].T)
    else:
        rxa[...] = np.copy(
            data["PIM_COND"].T + data["PIM_COND_LEAK"].T + data["nfa"].T
        )
    txa[...] = np.copy(data["txa"].T)
    signal_config = SignalConfig(data)
    bf_lengths = {"utd_nlin_mult_infl_fix_pwr": [16],
        "sep_nlin_mult_infl_fix_pwr": [16],
        "utd_nlin_fix_power": [1],
        "utd_nlin": [2]}
    result = multrans_model(rxa, txa, conv_data, bf_lengths[config.model],
                   config, signal_config)
    
    result_path = os.path.join(output_dir, config.poly, config.pim_type+"_pim/")
    os.makedirs(result_path, exist_ok=True)
    filename = data_marker +'_back_{}_fwd_{}.npz'.format(n_back, n_fwd)
    np.savez(os.path.join(result_path, filename), signal_dict=result)
