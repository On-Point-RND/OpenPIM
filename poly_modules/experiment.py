import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from itertools import product
from scipy.signal import convolve
from scipy.io import loadmat
import json

from gen_mat import *
from ls_model import *
from metrics import *

@dataclass
class SignalConfig:
    fs: float
    fc_tx: float
    pim_sft: float
    pim_bw: float
    pim_total_bw: float


@dataclass
class WindowExpConfig:
    n_back: list
    n_fwd: list
    bf_len: int
    model: str
    poly: str


class ModelExpConfig:
    def __init__(self, filename):
           with open(filename, 'r') as f:
            config = json.load(f)
            self.models = config['models']
            self.poly_list = config['poly_list']
            self.pim_type = config['pim_type']
            self.back_list = config['back_list']
            self.fwd_list = config['fwd_list']


def experiment(experiment_name, output_dir = './Results/Polynomial_experiments/',
                              data_path = '5m'):
    print('***************************************************************************************************')
    print(f"Running experiment: {experiment_name}")

    model_config = ModelExpConfig('config.json')

    os.makedirs(output_dir, exist_ok=True)
    multi_trans = False
    if data_path == '5m':
        data = loadmat("../Data/FOR_COOPERATION/1TR_C20Nc1CD_E20Ne1CD_20250117_5m/1TR_C20Nc1CD_E20Ne1CD_20250117_5m.mat")
    elif data_path == '0.5m':
        data = loadmat("../Data/FOR_COOPERATION/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m.mat")
    elif data_path == '1L':
        multi_trans = True
        data = loadmat("../Data/FOR_COOPERATION/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L.mat")
    elif data_path == '16L':
        multi_trans = True
        data = loadmat("../Data/FOR_COOPERATION/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L.mat")

    fil = loadmat("../Data/FOR_COOPERATION/rx_filter.mat")
    n,m = data["rxa"].shape[1], data["rxa"].shape[0] 
    conv_data = fil['flt_coeff'].flatten()
    rxa = np.empty((n,m), dtype=np.complex128, order='F')
    txa = np.empty((n,m), dtype=np.complex128, order='F')
    # nfa = np.empty((n,m), dtype=np.complex128, order='F')
    if model_config.pim_type == 'all':
        rxa[...] = np.copy(data["rxa"].T)
    elif model_config.pim_type == 'ext':
        rxa[...] = np.copy(data["PIM_EXT"].T + data["nfa"].T)
    else:
        rxa[...] = np.copy(
            data["PIM_COND"].T + data["PIM_COND_LEAK"].T + data["nfa"].T
        )
    txa[...] = np.copy(data["txa"].T)
    # nfa[...] = np.copy(data["nfa"].T)
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FC_RX = data['BANDS_UL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6
    PIM_SFT = data['PIM_sft'][0][0] / 10**6
    PIM_BW = data['BANDS_TX'][0][0][1][0][0] / 10**6
    PIM_total_BW = data['BANDS_TX'][0][0][3][0][0] / 10**6
    signal_config = SignalConfig(FS, FC_TX, PIM_SFT, PIM_BW, PIM_total_BW)
    model_config.bf_lengths = {"utd_nlin_mult_infl_fix_pwr": [16],
        "sep_nlin_mult_infl_fix_pwr": [16]}
    for model_name, poly_name in product(model_config.models, model_config.poly_list):
        print(model_name)
        train_gl, test_gl, params_gl = [], [], []
        signal_dict = dict()
        for bf_len in model_config.bf_lengths[model_name]:
            window_config = WindowExpConfig(model_config.back_list, model_config.fwd_list, bf_len, model_name, poly_name)
            if multi_trans:
                train_metrics, test_metrics, params = window_experiment_multr(rxa, txa, conv_data, window_config, signal_config, signal_dict)
            else:
                train_metrics, test_metrics, params = window_experiment(rxa, txa, conv_data, window_config, signal_config)
            train_gl += train_metrics
            test_gl += test_metrics
            params_gl += params
        result_path = os.path.join(output_dir,poly_name+"/")
        os.makedirs(result_path, exist_ok=True)
        pd.DataFrame({'Train_metric': train_gl, 'Test_metric': test_gl, 
                    'Back' : [v[0] for v in  params_gl], 'Forward' : [v[1] for v in params_gl], 
                    'Degree': [v[2] for v in  params_gl]}).to_csv(
                        result_path + data_path +'_{}_{}_metrics.tsv'.format(model_name, model_config.pim_type
                        ))
        if multi_trans:
            np.savez(result_path + data_path +'_signals.npz', signal_dict=signal_dict)
    return True


def window_experiment(rxa, txa, conv4metrics, config: WindowExpConfig, sig_config: SignalConfig):
    back_list, fwd_list, bf_len = config.n_back, config.n_fwd, config.bf_len
    fs, pim_sft, pim_bw = sig_config.fs, sig_config.pim_sft, sig_config.pim_bw
    model_func = globals()[config.model]
    poly_func = globals()[config.poly]
    n_back = max(back_list)
    n_fwd = max(fwd_list)
    n_train = int(rxa.shape[0] * 0.8)
    rxa_train_mem = rxa[:n_train].reshape(-1)
    txa_train_mem = txa[:n_train].reshape(-1)
    rxa_test_mem = rxa[n_train:].reshape(-1)
    txa_test_mem = txa[n_train:].reshape(-1)
    rxa_train = rxa_train_mem[n_back:-n_fwd]
    rxa_test = rxa_test_mem[n_back:-n_fwd]
    
    n_train = rxa_train.shape[0]
    n_test = rxa_test.shape[0]

    pred_train = np.empty((n_train,), dtype=np.complex128, order='F')
    pred_test = np.empty((n_test,), dtype=np.complex128, order='F')

    conv_train = np.empty((n_train+254,), dtype=np.complex128, order='F')
    conv_test = np.empty((n_test+254,), dtype=np.complex128, order='F')
    conv_train[...] = convolve(rxa_train.reshape(-1), conv4metrics)
    conv_test[...] = convolve(rxa_test.reshape(-1), conv4metrics)

    conv_pred_train = np.empty((n_train+254,), dtype=np.complex128, order='F')
    conv_pred_test = np.empty((n_test+254,), dtype=np.complex128, order='F')


    mmat_train = create_model_matrix(
        model_func, poly_func,
        txa_train_mem, bf_len, n_back, n_fwd
    )

    mmat_test = create_model_matrix(
        model_func, poly_func,
        txa_test_mem, bf_len, n_back, n_fwd
    )

    train_metrics = []
    test_metrics = []
    params = []
    m = mmat_train.shape[1]
    for i_back in back_list:
        for i_fwd in fwd_list:
            idx_start = (n_back - i_back) * bf_len
            idx_end = m - (n_fwd - i_fwd) * bf_len
            mmat_train_slice = mmat_train[:, idx_start:idx_end]
            mmat_test_slice = mmat_test[:, idx_start:idx_end]
            model_wts = ls_solve(mmat_train_slice, rxa_train)
            pred_train[...] = rxa_train - mmat_train_slice @ model_wts
            pred_test[...] = rxa_test - mmat_test_slice @ model_wts
            conv_pred_train = convolve(pred_train, conv4metrics)
            conv_pred_test = convolve(pred_test, conv4metrics)
            train_metric_value = calculate_metrics(
                conv_train, conv_pred_train, fs, pim_sft, pim_bw
            )
            test_metric_value = calculate_metrics(
                conv_test, conv_pred_test, fs, pim_sft, pim_bw)
            train_metrics.append(train_metric_value)
            test_metrics.append(test_metric_value)
            params.append([i_back, i_fwd, bf_len])
    return train_metrics, test_metrics, params


def window_experiment_multr(rxa, txa, conv4metrics, config: WindowExpConfig, sig_config: SignalConfig, signal_dict: dict):
    back_list, fwd_list, bf_len = config.n_back, config.n_fwd, config.bf_len
    fs, pim_sft, pim_bw = sig_config.fs, sig_config.pim_sft, sig_config.pim_bw
    model_func = globals()[config.model]
    poly_func = globals()[config.poly]
    n_back = max(back_list)
    n_fwd = max(fwd_list)
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
    print(rxa_train.shape, conv_train.shape, conv4metrics.shape)
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

    train_metrics = []
    test_metrics = []
    params = []
    m = mtn_train.shape[1]
    for i_back in back_list:
        for i_fwd in fwd_list:
            idx_start = (n_back - i_back) * bf_len
            idx_end = m - (n_fwd - i_fwd) * bf_len
            mtn_train_slice = mtn_train[:, idx_start:idx_end, :]
            mtn_test_slice = mtn_test[:, idx_start:idx_end, :]
            model_wts = ls_multi_trans(mtn_train_slice, rxa_train)
            pred_train[...] = rxa_train
            pred_test[...] = rxa_test
            print(model_wts.shape)
            contract(mtn_train_slice, model_wts, pred_train)
            contract(mtn_test_slice, model_wts, pred_test)
            convolve_tensor(pred_train, conv4metrics, conv_pred_train)
            convolve_tensor(pred_test, conv4metrics, conv_pred_test)
            train_metric_value = calculate_avg_metrics(
                conv_train, conv_pred_train, fs, pim_sft, pim_bw
            )
            test_metric_value = calculate_avg_metrics(
                conv_test, conv_pred_test, fs, pim_sft, pim_bw)
            signal_dict[(i_back, i_fwd, bf_len)] = [conv_test, conv_pred_test]
            train_metrics.append(train_metric_value)
            test_metrics.append(test_metric_value)
            params.append([i_back, i_fwd, bf_len])
    return train_metrics, test_metrics, params

if __name__ == '__main__':
    experiment('test', data_path = '16L')
