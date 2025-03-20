import numpy as np
from numpy.polynomial import Chebyshev as T
from numpy.polynomial import Legendre as L


def contract(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    n = z.shape[1]
    for i in range(n):
        z[:,i] -= x[:, :, i] @ y[:, i]
    return True


def cheb(x, n):
    """
    returns T_n(x)
    value of not normalized Chebyshev polynomial
    $\int \frac1{\sqrt{1-x^2}}T_m(x)T_n(x) dx = \frac\pi2\delta_{nm}$
    """
    return T.basis(n)(x)


def simple_model_tens(x: np.ndarray, model_mem: np.ndarray, n_tr: int):
    assert x.shape[0] == model_mem.shape[0]
    n_trans = x.shape[1]
    assert n_trans == model_mem.shape[2]
    for tr in range(n_trans):
        for loc_tr in range(n_trans):
            model_mem[:, loc_tr, tr] = np.copy(x[:, loc_tr] * (np.abs(x[:, loc_tr]) ** 2))
    return True


def cheb_model_tens(x: np.ndarray, model_mem: np.ndarray, n_tr: int):
    assert x.shape[0] == model_mem.shape[0]
    pow = int(model_mem.shape[1] / n_tr)
    for tr in range(n_tr):
        for loc_tr in range(n_tr):
            for i in range(pow):
                model_mem[:, loc_tr*pow+i, tr] = x[:, loc_tr] * cheb(np.abs(x[:, loc_tr]), i)
    return True


def create_model_tensor(x: np.ndarray, model,
                        n_bf: int, n_back: int, n_fwd: int):
    n_trans = x.shape[1]
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    x_work_range = x[n_back : -n_fwd]
    assert x_work_range.shape[0] == n_pts
    model_mat = np.empty((n_pts, n_bf*win_len, n_trans), dtype=np.complex128, order='F')
    for i in range(win_len):
        model(x[i:n_pts+i], model_mat[:,i*n_bf:(i+1)*n_bf, :], n_trans)
    return model_mat
