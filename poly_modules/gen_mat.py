import numpy as np
from numpy.polynomial import Chebyshev as T
from numpy.polynomial import Legendre as L


def cheb(x, n):
    """
    returns T_n(x)
    value of not normalized Chebyshev polynomial
    $\int \frac1{\sqrt{1-x^2}}T_m(x)T_n(x) dx = \frac\pi2\delta_{nm}$
    """
    return T.basis(n)(x)


def legendre(x, n, interval=(-1.0, 1.0)):
    """
    Non-normed poly
    """
    xn = (interval[0] + interval[1] - 2.0*x)/(interval[0] - interval[1])
    return L.basis(n)(xn)


def simple_model(x: np.ndarray, model_mem: np.ndarray):
    assert x.shape[0] == model_mem.shape[0]
    model_mem[:,0] = np.copy(x * (np.abs(x) ** 2))
    return True


def cheb_model(x: np.ndarray, model_mem: np.ndarray):
    assert x.shape[0] == model_mem.shape[0]
    pow = model_mem.shape[1]
    for i in range(pow):
        model_mem[:, i] = x * cheb(np.abs(x), i)
    return True


def legendre_model(x: np.ndarray, model_mem: np.ndarray):
    assert x.shape[0] == model_mem.shape[0]
    pow = model_mem.shape[1]
    for i in range(pow):
        model_mem[:, i] = x * legendre(np.abs(x), i)
    return True


def create_model_matrix(x: np.ndarray, model,
                        n_bf: int, n_back: int, n_fwd: int):
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    x_work_range = x[n_back : -n_fwd]
    assert x_work_range.shape[0] == n_pts
    model_mat = np.empty((n_pts, n_bf*win_len), dtype=np.complex128, order='F')
    for i in range(win_len):
        model(x[i:n_pts+i], model_mat[:,i*n_bf:(i+1)*n_bf])
    return model_mat


if __name__ == '__main__':
    x = np.arange(30)
    matrix = create_model_matrix(x, simple_model, 1, 5, 2)
    print(matrix)
