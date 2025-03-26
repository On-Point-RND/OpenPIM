import numpy as np
from basis_func import *


def create_model_matrix(model_func: Callable[..., bool],
                        poly_func: Callable[..., np.ndarray],
                        x: np.ndarray, n_bf: int, n_back: int, n_fwd: int):
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    x_work_range = x[n_back : -n_fwd]
    assert x_work_range.shape[0] == n_pts
    model_mat = np.empty((n_pts, n_bf*win_len), dtype=np.complex128, order='F')
    for i in range(win_len):
        model_func(
            poly_func,x[i:n_pts+i],
            model_mat[:,i*n_bf:(i+1)*n_bf]
        )
    return model_mat


def create_model_tensor(model_func: Callable[..., bool],
                        poly_func: Callable[..., np.ndarray],
                        x: np.ndarray, n_bf: int, n_back: int, n_fwd: int):
    assert len(x.shape) > 1
    n_trans = x.shape[1]
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    x_work_range = x[n_back : -n_fwd]
    assert x_work_range.shape[0] == n_pts
    model_mat = np.empty(
        (n_pts, n_bf*win_len, n_trans), dtype=np.complex128, order='F'
    )
    for i_ts in range(n_trans):
        for i in range(win_len):
            model_func(
                poly_func,
                x[i:n_pts+i], model_mat[:,i*n_bf:(i+1)*n_bf, i_ts],
                i_ts
            )
    return model_mat


if __name__ == '__main__':
    x = np.arange(30)
    matrix = create_model_matrix(
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
