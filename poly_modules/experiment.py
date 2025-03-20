import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from scipy.signal import convolve
from scipy.io import loadmat

from gen_mat import *
from gen_tens import *
from ls_model import *
from ..pim_utils.pim_metrics import *

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


@dataclass
class ModelExpConfig:
    bf_lengths: dict
    models: list


def model_experiment(experiment_name, output_dir = '../Results/Polynomial_experiments/',
                              data_path = '5m'):
    print('***************************************************************************************************')
    print(f"Running experiment: {experiment_name}")

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
    nfa = np.empty((n,m), dtype=np.complex128, order='F')
    rxa[...] = np.copy(data["rxa"].T)
    txa[...] = np.copy(data["txa"].T)
    nfa[...] = np.copy(data["nfa"].T)
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FC_RX = data['BANDS_UL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6
    PIM_SFT = data['PIM_sft'][0][0] / 10**6
    PIM_BW = data['BANDS_TX'][0][0][1][0][0] / 10**6
    PIM_total_BW = data['BANDS_TX'][0][0][3][0][0] / 10**6
    signal_config = SignalConfig(FS, FC_TX, PIM_SFT, PIM_BW, PIM_total_BW)

    bf_dict_str = {'simple_model': [1],
               'cheb_model': [1,2],
               'legendre_model': [1,2]}
    bf_dict_multr = {'simple_model_tens': [16],
               'cheb_model_tens': [16,32]}
    back_list, fwd_list = [], []
    if multi_trans:
        model_config = ModelExpConfig(bf_dict_multr, ['cheb_model_tens'])
        back_list = np.arange(1,3,1).tolist()
        fwd_list = np.arange(2).tolist()
    else:
        model_config = ModelExpConfig(bf_dict_str, ['simple_model'])
        back_list = np.arange(10,30,1).tolist()
        fwd_list = np.arange(5).tolist()

    for model_name in model_config.models:
        print(model_name)
        train_gl, test_gl, params_gl = [], [], []
        signal_dict = dict()
        for bf_len in model_config.bf_lengths[model_name]:
            window_config = WindowExpConfig(back_list, fwd_list, bf_len, model_name)
            if multi_trans:
                train_metrics, test_metrics, params = window_experiment_multr(rxa, txa, conv_data, window_config, signal_config, signal_dict)
            else:
                train_metrics, test_metrics, params = window_experiment(rxa, txa, conv_data, window_config, signal_config)
            train_gl += train_metrics
            test_gl += test_metrics
            params_gl += params
        result_path = os.path.join(output_dir,model_name.split("_")[0]+"/")
        os.makedirs(result_path, exist_ok=True)
        pd.DataFrame({'Train_metric': train_gl, 'Test_metric': test_gl, 
                    'Back' : [v[0] for v in  params_gl], 'Forward' : [v[1] for v in params_gl], 
                    'Degree': [v[2] for v in  params_gl]}).to_csv(result_path + data_path +'_polynomial_metrics.tsv')
        if multi_trans:
            np.savez(result_path + data_path +'_signals.npz', signal_dict=signal_dict)
    return True


def window_experiment(rxa, txa, conv4metrics, config: WindowExpConfig, sig_config: SignalConfig):
    back_list, fwd_list, bf_len = config.n_back, config.n_fwd, config.bf_len
    fs, pim_sft, pim_bw = sig_config.fs, sig_config.pim_sft, sig_config.pim_bw
    model = globals()[config.model]
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


    mmat_train = create_model_matrix(txa_train_mem, model, bf_len,
                               n_back, n_fwd)
    mmat_test = create_model_matrix(txa_test_mem, model, bf_len,
                               n_back, n_fwd)

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
    model = globals()[config.model]
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
    convolve_tensor(rxa_train, conv_train, conv4metrics)
    convolve_tensor(rxa_test, conv_test, conv4metrics)

    conv_pred_train = np.empty((n_train+254,n_trans), dtype=np.complex128, order='F')
    conv_pred_test = np.empty((n_test+254,n_trans), dtype=np.complex128, order='F')

    mtn_train = create_model_tensor(txa_train_mem, model, bf_len,
                               n_back, n_fwd)
    mtn_test = create_model_tensor(txa_test_mem, model, bf_len,
                               n_back, n_fwd)

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
            convolve_tensor(pred_train, conv_pred_train, conv4metrics)
            convolve_tensor(pred_test, conv_pred_test, conv4metrics)
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
    model_experiment('test', data_path = '16L')
