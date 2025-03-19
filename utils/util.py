import numpy as np


def count_net_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes
    return n_param


def get_amplitude(IQ_signal):
    I = IQ_signal[:, 0]
    Q = IQ_signal[:, 1]
    power = I ** 2 + Q ** 2
    amplitude = np.sqrt(power)
    return amplitude

