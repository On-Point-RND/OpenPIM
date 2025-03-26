import numpy as np
from scipy.signal import convolve

from gen_mat import *


def convolve_tensor(x: np.ndarray, filter: np.ndarray, x_conv: np.ndarray):
    for i in range(x.shape[1]):
        x_conv[:, i] = convolve(x[:,i], filter)
    return True


def contract(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    n = z.shape[1]
    for i in range(n):
        z[:,i] -= x[:, :, i] @ y[:, i]
    return True


def ls_multi_trans(model_tens: np.ndarray, rhs: np.ndarray):
    assert model_tens.shape[0] == rhs.shape[0]
    assert model_tens.shape[2] == rhs.shape[1]
    n_trans = rhs.shape[1]
    wts_tens = np.empty((model_tens.shape[1], rhs.shape[1]), dtype=np.complex128)
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


def ls_solve(model_mat: np.ndarray, rhs: np.ndarray):
    assert model_mat.shape[0] == rhs.shape[0]
    wts = np.empty((model_mat.shape[1],), dtype=np.complex128)
    inverted = model_mat.conj().T @ model_mat
    if np.linalg.cond(inverted) > 10**5:
        I = np.eye(inverted.shape[0], dtype=np.complex128)
        psi = 10**(-10)
        wts = np.linalg.inv(inverted + I*psi) @ model_mat.conj().T @ rhs
    else:
        wts = np.linalg.inv(inverted) @ model_mat.conj().T @ rhs
    return wts
