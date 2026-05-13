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


def volterra2_tensor_feature_count(
    n_trans: int,
    win_len: int,
    volterra_include_quadratic: bool = True,
    volterra_include_conjugate: bool = False,
    volterra_include_abs: bool = False,
):
    n_linear = n_trans * win_len
    n_quadratic = (
        n_linear * (n_linear + 1) // 2 if volterra_include_quadratic else 0
    )
    n_conj_quadratic = n_linear * n_linear if volterra_include_conjugate else 0
    n_abs_quadratic = n_linear * n_linear if volterra_include_abs else 0
    return 1 + n_linear + n_quadratic + n_conj_quadratic + n_abs_quadratic


def create_volterra2_tensor(
    x: np.ndarray,
    n_back: int,
    n_fwd: int,
    volterra_include_quadratic: bool = True,
    volterra_include_conjugate: bool = False,
    volterra_include_abs: bool = False,
):
    assert len(x.shape) > 1
    n_trans = x.shape[1]
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    end_idx = -n_fwd if n_fwd > 0 else None
    x_work_range = x[n_back:end_idx]
    assert x_work_range.shape[0] == n_pts

    n_linear = n_trans * win_len
    n_features = volterra2_tensor_feature_count(
        n_trans,
        win_len,
        volterra_include_quadratic=volterra_include_quadratic,
        volterra_include_conjugate=volterra_include_conjugate,
        volterra_include_abs=volterra_include_abs,
    )
    linear_mat = np.empty((n_pts, n_linear), dtype=np.complex128, order='F')
    feature_mat = np.empty((n_pts, n_features), dtype=np.complex128, order='F')
    feature_mat[:, 0] = 1.0

    idx = 0
    for lag in range(win_len):
        x_lag = x[lag:n_pts + lag]
        for i_tr in range(n_trans):
            linear_mat[:, idx] = x_lag[:, i_tr]
            idx += 1
    assert idx == n_linear
    feature_mat[:, 1:1 + n_linear] = linear_mat

    idx = 1 + n_linear
    if volterra_include_quadratic:
        for i_feature in range(n_linear):
            for j_feature in range(i_feature, n_linear):
                feature_mat[:, idx] = (
                    linear_mat[:, i_feature] * linear_mat[:, j_feature]
                )
                idx += 1

    if volterra_include_conjugate:
        for i_feature in range(n_linear):
            for j_feature in range(n_linear):
                feature_mat[:, idx] = (
                    linear_mat[:, i_feature] * np.conj(linear_mat[:, j_feature])
                )
                idx += 1

    if volterra_include_abs:
        for i_feature in range(n_linear):
            for j_feature in range(n_linear):
                feature_mat[:, idx] = (
                    linear_mat[:, i_feature] * np.abs(linear_mat[:, j_feature])
                )
                idx += 1

    assert idx == n_features
    tens = np.empty((n_pts, n_features, n_trans), dtype=np.complex128, order='F')
    for i_ts in range(n_trans):
        tens[:, :, i_ts] = feature_mat
    return tens


def ls_solve(
    model_tens: np.ndarray,
    rhs: np.ndarray,
    verbose_basis_fit: bool = False,
):
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
        if verbose_basis_fit:
            pred_i = model_mat @ wts_tens[:, i_tr]
            p_rx = float(np.mean(np.abs(rhs_i) ** 2))
            nmse = float(np.mean(np.abs(rhs_i - pred_i) ** 2)) / max(p_rx, 1e-30)
            rho = np.abs(np.vdot(rhs_i, pred_i)) / (
                np.linalg.norm(rhs_i) * np.linalg.norm(pred_i) + 1e-30
            )
            nmse_db = 10 * np.log10(nmse) if nmse > 0 else -np.inf
            if nmse_db > -0.5 and rho < 0.1:
                hint = (
                    "Weak fit: NMSE≈0 dB and |corr|≈0; "
                    "RX is almost outside the linear span of X columns."
                )
            else:
                hint = "The feature-basis fit looks meaningful."
            print(
                "[ls_solve] RX approximation in feature basis (train): "
                f"channel {i_tr}: NMSE_lin={nmse_db:.2f} dB; "
                f"|corr|(RX, Xw)={rho:.4f}. {hint}"
            )
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
