import numpy as np
from typing import Callable, Any

from numpy.polynomial import Chebyshev as T
from numpy.polynomial import Legendre as L
from numpy.polynomial import Polynomial as P


def power(x, n):
    return P.basis(n)(x)


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


def poly_fix_power(poly_func: Callable[..., np.ndarray],
                   x: np.ndarray, mem_segment: np.ndarray):
    assert x.shape[0] == mem_segment.shape[0]
    mem_segment[:,0] = x * poly_func(np.abs(x), 2)
    return True


def poly_model(poly_func: Callable[..., np.ndarray],
               x: np.ndarray, mem_segment: np.ndarray):
    assert x.shape[0] == mem_segment.shape[0]
    pow = mem_segment.shape[1]
    for i in range(pow):
        mem_segment[:, i] = x * poly_func(np.abs(x), i)
    return True



def simple_mult_infl(poly_func: Callable[..., np.ndarray],
                     x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    n_trans = x.shape[1]
    for i_tr in range(n_trans):
        mem_segment[:, i_tr] = x[:, i_tr] * poly_func(np.abs(x[:, i_tr]), 2)
    return True


def sep_nlin_mult_infl(poly_func: Callable[..., np.ndarray],
                       x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    n_ts = x.shape[1]
    pow = int(mem_segment.shape[1] / n_ts)
    for i_tr in range(n_ts):
        for i in range(pow):
            mem_segment[:, i_tr*pow+i] = x[:, i_tr] * poly_func(np.abs(x[:, i_tr]), i)
    return True


def sep_nlin_mult_infl_fix_pwr(poly_func: Callable[..., np.ndarray],
                       x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    n_ts = x.shape[1]
    pow = int(mem_segment.shape[1] / n_ts)
    for i_tr in range(n_ts):
        mem_segment[:, i_tr] = x[:, i_tr] * poly_func(np.abs(x[:, i_tr]), 2)
    return True


def utd_nlin(poly_func: Callable[..., np.ndarray],
             x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    pow = int(mem_segment.shape[1])
    total_x = np.sum(x, axis=1)
    for i in range(pow):
        mem_segment[:, i] = x[:, ts] * poly_func(np.abs(total_x), i)
    return True


def utd_nlin_mult_infl(poly_func: Callable[..., np.ndarray],
                       x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    n_ts = x.shape[1]
    pow = int(mem_segment.shape[1] / n_ts)
    total_x = np.sum(x, axis=1)
    for i_tr in range(n_ts):
        for i in range(pow):
            mem_segment[:, i_tr*pow+i] = x[:, i_tr] * poly_func(np.abs(total_x), i)
    return True


def utd_nlin_fix_power(poly_func: Callable[..., np.ndarray],
             x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    total_x = np.sum(x, axis=1)
    mem_segment[:, 0] = x[:, ts] * poly_func(np.abs(total_x), 2)
    return True


def utd_nlin_mult_infl_fix_pwr(poly_func: Callable[..., np.ndarray],
                       x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    n_ts = x.shape[1]
    total_x = np.sum(x, axis=1)
    for i_tr in range(n_ts):
        mem_segment[:, i_tr] = x[:, i_tr] * poly_func(np.abs(total_x), 2)
    return True


def utd_nlin_self_act(poly_func: Callable[..., np.ndarray],
                      x: np.ndarray, mem_segment: np.ndarray, ts: int):
    assert x.shape[0] == mem_segment.shape[0]
    pow = int(mem_segment.shape[1])
    assert 2*pow == mem_segment.shape[1]
    total_x = np.sum(x, axis=1)
    for i in range(pow):
        mem_segment[:, i] = x[:, ts] * poly_func(np.abs(total_x), i)
        mem_segment[:, pow+i] = x[:, ts] * poly_func(np.abs(x[:, ts]), i)
    return True
