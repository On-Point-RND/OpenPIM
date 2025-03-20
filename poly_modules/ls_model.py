import numpy as np
from scipy.signal import convolve

from gen_mat import *
from gen_tens import *


def convolve_tensor(x: np.ndarray, x_conv: np.ndarray, filter: np.ndarray):
    for i in range(x.shape[1]):
        x_conv[:, i] = convolve(x[:,i], filter)
    return True


def ls_multi_trans(model_tens: np.ndarray, rhs_tens: np.ndarray):
    assert model_tens.shape[0] == rhs_tens.shape[0]
    assert model_tens.shape[2] == rhs_tens.shape[1]
    n_trans = rhs_tens.shape[1]
    wts_tens = np.empty((model_tens.shape[1], rhs_tens.shape[1]), dtype=np.complex128)
    for i_tr in range(n_trans):
        rhs = rhs_tens[:, i_tr]
        model_mat = model_tens[:,:,i_tr]
        inverted = model_mat.conj().T @ model_mat
        tmp_prod = model_mat.conj().T @ rhs
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
