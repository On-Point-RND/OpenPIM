from typing import Callable

import numpy as np
from scipy.signal import convolve
from basis_func import *


def convolve_tensor(x: np.ndarray, filter: np.ndarray, x_conv: np.ndarray):
    for i in range(x.shape[1]):
        x_conv[:, i] = convolve(x[:,i], filter)
    return True


def contract(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    n = z.shape[1]
    for i in range(n):
        z[:,i] = x[:, :, i] @ y[:, i]
    return True


def create_model_tensor(model_func: Callable[..., bool],
                        poly_func: Callable[..., np.ndarray],
                        x: np.ndarray, n_bf: int, n_back: int, n_fwd: int):
    assert len(x.shape) > 1
    n_trans = x.shape[1]
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    # n_fwd==0: x[n_back:-0] is x[n_back:0] (empty); use None to mean "to end"
    end_idx = -n_fwd if n_fwd > 0 else None
    x_work_range = x[n_back:end_idx]
    assert x_work_range.shape[0] == n_pts
    tens = np.empty(
        (n_pts, n_bf*win_len, n_trans), dtype=np.complex128, order='F'
    )
    for i_ts in range(n_trans):
        for i in range(win_len):
            model_func(
                poly_func,
                x[i:n_pts+i], tens[:,i*n_bf:(i+1)*n_bf, i_ts],
                i_ts
            )
    return tens


def volterra2_tensor_feature_count(n_trans: int, win_len: int):
    n_linear = n_trans * win_len
    n_q_channels = n_trans * n_trans
    n_lag_pairs = win_len * (win_len + 1) // 2
    return n_linear + n_q_channels * n_lag_pairs


def create_volterra2_tensor(x: np.ndarray, n_back: int, n_fwd: int):
    assert len(x.shape) > 1
    n_trans = x.shape[1]
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    end_idx = -n_fwd if n_fwd > 0 else None
    x_work_range = x[n_back:end_idx]
    assert x_work_range.shape[0] == n_pts

    n_features = volterra2_tensor_feature_count(n_trans, win_len)
    feature_mat = np.empty((n_pts, n_features), dtype=np.complex128, order='F')

    idx = 0
    for lag in range(win_len):
        x_lag = x[lag:n_pts + lag]
        for i_tr in range(n_trans):
            feature_mat[:, idx] = x_lag[:, i_tr]
            idx += 1

    for lag_a in range(win_len):
        x_lag_a = x[lag_a:n_pts + lag_a]
        for lag_b in range(lag_a, win_len):
            x_lag_b = x[lag_b:n_pts + lag_b]
            for i_tr in range(n_trans):
                for j_tr in range(n_trans):
                    feature_mat[:, idx] = (
                        x_lag_a[:, i_tr] * x_lag_b[:, j_tr]
                    )
                    idx += 1

    assert idx == n_features
    tens = np.empty((n_pts, n_features, n_trans), dtype=np.complex128, order='F')
    for i_ts in range(n_trans):
        tens[:, :, i_ts] = feature_mat
    return tens


def ls_solve(model_tens: np.ndarray, rhs: np.ndarray):
    assert model_tens.shape[0] == rhs.shape[0]
    assert model_tens.shape[2] == rhs.shape[1]
    n_trans = rhs.shape[1]
    wts_tens = np.empty((model_tens.shape[1], n_trans), dtype=np.complex128)
    for i_tr in range(n_trans):
        rhs_i = rhs[:, i_tr]
        model_mat = model_tens[:,:,i_tr]
        inverted = model_mat.conj().T @ model_mat
        tmp_prod = model_mat.conj().T @ rhs_i
        if np.linalg.cond(inverted) > 10**5:
            I = np.eye(inverted.shape[0], dtype=np.complex128)
            psi = 10**(-11)
            wts_tens[:, i_tr] = np.linalg.inv(inverted + I*psi) @ tmp_prod
        else:
            wts_tens[:, i_tr] = np.linalg.inv(inverted) @ tmp_prod
    return wts_tens


if __name__ == '__main__':
    x = np.random.rand(30, 1)
    matrix = create_model_tensor(
        poly_fix_power,
        cheb,
        x, 1, 5, 2)
    print(matrix)

    x = np.random.rand(30, 2)
    tens = create_model_tensor(
        simple_mult_infl,
        power,
        x,
        2, 2, 2
    )
    print(tens)
    print(tens.shape)
