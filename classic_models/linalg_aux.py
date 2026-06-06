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


VOLTERRA_FULL_MODELS = frozenset({"volterra_full"})


def _validate_volterra_order(volterra_order: int):
    if volterra_order not in (2, 3):
        raise ValueError("volterra_order must be 2 or 3")


def is_volterra_full_model(model: str) -> bool:
    return model in VOLTERRA_FULL_MODELS


def volterra_tensor_feature_count(
    n_trans: int,
    win_len: int,
    volterra_order: int = 2,
    volterra_include_quadratic: bool = True,
    volterra_include_conjugate: bool = False,
    volterra_include_abs: bool = False,
    volterra_include_abs_sq: bool = False,
    volterra_include_cubic: bool = False,
    volterra_include_cubic_conj: bool = True,
    volterra_include_cubic_abs: bool = False,
    volterra_include_cubic_abs_sq: bool = False,
):
    _validate_volterra_order(volterra_order)
    n_linear = n_trans * win_len
    n_features = 1 + n_linear
    if volterra_order >= 2:
        if volterra_include_quadratic:
            n_features += n_linear * (n_linear + 1) // 2
        if volterra_include_conjugate:
            n_features += n_linear * n_linear
        if volterra_include_abs:
            n_features += n_linear * n_linear
        if volterra_include_abs_sq:
            n_features += n_linear * n_linear
    if volterra_order >= 3:
        if volterra_include_cubic:
            n_features += n_linear * (n_linear + 1) * (n_linear + 2) // 6
        if volterra_include_cubic_conj:
            n_features += n_linear ** 3
        if volterra_include_cubic_abs:
            n_features += n_linear ** 3
        if volterra_include_cubic_abs_sq:
            n_features += n_linear ** 3
    return n_features


def _build_volterra_linear_mat(
    x: np.ndarray, n_back: int, n_fwd: int
):
    assert len(x.shape) > 1
    n_trans = x.shape[1]
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    end_idx = -n_fwd if n_fwd > 0 else None
    x_work_range = x[n_back:end_idx]
    assert x_work_range.shape[0] == n_pts

    n_linear = n_trans * win_len
    linear_mat = np.empty((n_pts, n_linear), dtype=np.complex128, order='F')
    idx = 0
    for lag in range(win_len):
        x_lag = x[lag:n_pts + lag]
        for i_tr in range(n_trans):
            linear_mat[:, idx] = x_lag[:, i_tr]
            idx += 1
    assert idx == n_linear
    return linear_mat, n_pts, n_linear, n_trans


def _append_volterra_order2_blocks(
    feature_mat: np.ndarray,
    linear_mat: np.ndarray,
    idx: int,
    n_linear: int,
    volterra_include_quadratic: bool,
    volterra_include_conjugate: bool,
    volterra_include_abs: bool,
    volterra_include_abs_sq: bool,
) -> int:
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

    if volterra_include_abs_sq:
        for i_feature in range(n_linear):
            for j_feature in range(n_linear):
                feature_mat[:, idx] = (
                    np.abs(linear_mat[:, i_feature])
                    * np.abs(linear_mat[:, j_feature])
                )
                idx += 1
    return idx


def _append_volterra_order3_blocks(
    feature_mat: np.ndarray,
    linear_mat: np.ndarray,
    idx: int,
    n_linear: int,
    volterra_include_cubic: bool,
    volterra_include_cubic_conj: bool,
    volterra_include_cubic_abs: bool,
    volterra_include_cubic_abs_sq: bool,
) -> int:
    if volterra_include_cubic:
        for i_feature in range(n_linear):
            for j_feature in range(i_feature, n_linear):
                for k_feature in range(j_feature, n_linear):
                    feature_mat[:, idx] = (
                        linear_mat[:, i_feature]
                        * linear_mat[:, j_feature]
                        * linear_mat[:, k_feature]
                    )
                    idx += 1

    if volterra_include_cubic_conj:
        for i_feature in range(n_linear):
            for j_feature in range(n_linear):
                for k_feature in range(n_linear):
                    feature_mat[:, idx] = (
                        linear_mat[:, i_feature]
                        * linear_mat[:, j_feature]
                        * np.conj(linear_mat[:, k_feature])
                    )
                    idx += 1

    if volterra_include_cubic_abs:
        for i_feature in range(n_linear):
            for j_feature in range(n_linear):
                for k_feature in range(n_linear):
                    feature_mat[:, idx] = (
                        linear_mat[:, i_feature]
                        * linear_mat[:, j_feature]
                        * np.abs(linear_mat[:, k_feature])
                    )
                    idx += 1

    if volterra_include_cubic_abs_sq:
        for i_feature in range(n_linear):
            for j_feature in range(n_linear):
                for k_feature in range(n_linear):
                    feature_mat[:, idx] = (
                        linear_mat[:, i_feature]
                        * np.abs(linear_mat[:, j_feature])
                        * np.abs(linear_mat[:, k_feature])
                    )
                    idx += 1
    return idx


def create_volterra_tensor(
    x: np.ndarray,
    n_back: int,
    n_fwd: int,
    volterra_order: int = 2,
    volterra_include_quadratic: bool = True,
    volterra_include_conjugate: bool = False,
    volterra_include_abs: bool = False,
    volterra_include_abs_sq: bool = False,
    volterra_include_cubic: bool = False,
    volterra_include_cubic_conj: bool = True,
    volterra_include_cubic_abs: bool = False,
    volterra_include_cubic_abs_sq: bool = False,
):
    _validate_volterra_order(volterra_order)
    linear_mat, n_pts, n_linear, n_trans = _build_volterra_linear_mat(
        x, n_back, n_fwd
    )
    n_features = volterra_tensor_feature_count(
        n_trans,
        n_back + n_fwd + 1,
        volterra_order=volterra_order,
        volterra_include_quadratic=volterra_include_quadratic,
        volterra_include_conjugate=volterra_include_conjugate,
        volterra_include_abs=volterra_include_abs,
        volterra_include_abs_sq=volterra_include_abs_sq,
        volterra_include_cubic=volterra_include_cubic,
        volterra_include_cubic_conj=volterra_include_cubic_conj,
        volterra_include_cubic_abs=volterra_include_cubic_abs,
        volterra_include_cubic_abs_sq=volterra_include_cubic_abs_sq,
    )
    if n_features > 50_000:
        print(
            f"[volterra] warning: feature count {n_features} is large; "
            "LS solve may be slow or ill-conditioned."
        )

    feature_mat = np.empty((n_pts, n_features), dtype=np.complex128, order='F')
    feature_mat[:, 0] = 1.0
    feature_mat[:, 1:1 + n_linear] = linear_mat

    idx = 1 + n_linear
    if volterra_order >= 2:
        idx = _append_volterra_order2_blocks(
            feature_mat,
            linear_mat,
            idx,
            n_linear,
            volterra_include_quadratic,
            volterra_include_conjugate,
            volterra_include_abs,
            volterra_include_abs_sq,
        )
    if volterra_order >= 3:
        idx = _append_volterra_order3_blocks(
            feature_mat,
            linear_mat,
            idx,
            n_linear,
            volterra_include_cubic,
            volterra_include_cubic_conj,
            volterra_include_cubic_abs,
            volterra_include_cubic_abs_sq,
        )

    assert idx == n_features
    tens = np.empty((n_pts, n_features, n_trans), dtype=np.complex128, order='F')
    for i_ts in range(n_trans):
        tens[:, :, i_ts] = feature_mat
    return tens


def volterra2_tensor_feature_count(
    n_trans: int,
    win_len: int,
    volterra_include_quadratic: bool = True,
    volterra_include_conjugate: bool = False,
    volterra_include_abs: bool = False,
    **kwargs,
):
    return volterra_tensor_feature_count(
        n_trans,
        win_len,
        volterra_order=2,
        volterra_include_quadratic=volterra_include_quadratic,
        volterra_include_conjugate=volterra_include_conjugate,
        volterra_include_abs=volterra_include_abs,
        **kwargs,
    )


def create_volterra2_tensor(
    x: np.ndarray,
    n_back: int,
    n_fwd: int,
    volterra_include_quadratic: bool = True,
    volterra_include_conjugate: bool = False,
    volterra_include_abs: bool = False,
    **kwargs,
):
    return create_volterra_tensor(
        x,
        n_back,
        n_fwd,
        volterra_order=2,
        volterra_include_quadratic=volterra_include_quadratic,
        volterra_include_conjugate=volterra_include_conjugate,
        volterra_include_abs=volterra_include_abs,
        **kwargs,
    )


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
