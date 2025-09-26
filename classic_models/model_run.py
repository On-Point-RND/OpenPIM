import numpy as np
import os
from scipy.io import loadmat
import pyrallis
from config import Config

from linalg_aux import *
from classical_metrics import *


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


def model(rxa, txa, nfa, bf_len: int,
          config: Config):
    model_func = globals()[config.model]
    poly_func = globals()[config.poly]
    n_back = config.n_back
    n_fwd = config.n_fwd
    n_cut_train = int(rxa.shape[0] * config.train_ratio)
    n_cut_test = int(rxa.shape[0] * config.test_ratio)
    n_trans = rxa.shape[1]
    rxa_train_mem = rxa[:n_cut_train]
    txa_train_mem = txa[:n_cut_train]
    rxa_test_mem = rxa[n_cut_test:]
    txa_test_mem = txa[n_cut_test:]
    nfa_test_mem = nfa[n_cut_test:]
    rxa_train = rxa_train_mem[n_back:-n_fwd]
    rxa_test = rxa_test_mem[n_back:-n_fwd]
    nfa_test = nfa_test_mem[n_back:-n_fwd]
    n_test = rxa_test.shape[0]

    pred_test = np.empty(
        (n_test,n_trans), dtype=np.complex128, order='F'
    )
    mtn_train = create_model_tensor(
        model_func, poly_func,
        txa_train_mem, bf_len, n_back, n_fwd
    )
    mtn_test = create_model_tensor(
        model_func, poly_func,
        txa_test_mem,bf_len, n_back, n_fwd
    )
    model_wts = ls_solve(mtn_train, rxa_train)
    contract(mtn_test, model_wts, pred_test)
    return rxa_test, pred_test, nfa_test


def train_poly_model(config: Config):
    print('*****************************************************************')
    print(f"Running model: {config.model}")

    # Create output directory
    os.makedirs(config.log_out_dir, exist_ok=True)
    result_path = os.path.join(
        config.log_out_dir, config.model,
        config.dataset_name, f"{config.PIM_type}_pim/"
    )
    os.makedirs(result_path, exist_ok=True)

    # Load data
    data_path = os.path.join(
        config.dataset_path, config.dataset_name, config.dataset_name + ".mat"
    )
    data = loadmat(data_path)
    signal_config = SignalConfig(data)
    fs, pim_sft = (
        signal_config.fs, signal_config.pim_sft
    )
    pim_bw = signal_config.pim_bw
    filter = loadmat(config.filter_path)["flt_coeff"]

    n, m = data["rxa"].shape[1], data["rxa"].shape[0]
    rxa = np.empty((n,m), dtype=np.complex128, order='F')
    txa = np.empty((n,m), dtype=np.complex128, order='F')
    nfa = np.empty((n,m), dtype=np.complex128, order='F')

    # Select PIM type
    if config.PIM_type == 'total':
        rxa[...] = np.copy(data["rxa"].T)
    elif config.PIM_type == 'ext':
        rxa[...] = np.copy(data["PIM_EXT"].T + data["nfa"].T)
    elif config.PIM_type == 'cond':
        rxa[...] = np.copy(data["PIM_COND"].T + data["nfa"].T)
    elif config.PIM_type == 'leak':
        rxa[...] = np.copy(data["PIM_COND_LEAK"].T + data["nfa"].T)
    else:
        raise ValueError(f"Invalid PIM type: {config.PIM_type}")

    txa[...] = np.copy(data["txa"].T)
    nfa[...] = np.copy(data["nfa"].T)

    # Model memory lengths
    bf_lengths = {
        "combi_nlin_mult_infl_fix_pwr": [32],
        "sep_nlin_mult_infl_fix_pwr": [16],
        "sep_nlin_mult_infl": [48],
        "sep_nlin_mult_infl_cross_b": [32],
        "utd_nlin_mult_infl": [48],
        "utd_nlin_mult_infl_fix_pwr": [16],
        "utd_nlin_mult_infl_cross_a": [32],
        "utd_nlin_mult_infl_cross_b": [32],
        "utd_nlin_self_infl_fix_pwr": [1],
        "utd_nlin_self_infl": [2],
        "poly_fix_power": [1],
        "poly_series": [3]
    }

    bf_dim = bf_lengths[config.model][0]
    rxa_test, pred_test, nfa_test = model(rxa, txa, nfa, bf_dim, config)
    logs = calculate_metrics(
        pred_test, rxa_test, filter, fs,
        pim_sft, pim_bw
    )
    powers = dict()
    for key, value in (
        ("gt", rxa_test), ("err", rxa_test - pred_test), ("noise", nfa_test)
    ):
        powers[key] = [
            compute_power(value[:, id], fs, pim_sft, pim_bw)
            for id in range(value.shape[1])
        ]
    reduction_levels = list(logs["Reduction_level"].values())
    mean_red_level = sum(reduction_levels) / len(reduction_levels)
    max_red_level = max(reduction_levels)
    plot_total_perf(powers, max_red_level, mean_red_level, result_path)


if __name__ == '__main__':
    # Parse configuration from command line or config file
    config = pyrallis.parse(config_class=Config)
    train_poly_model(config)
