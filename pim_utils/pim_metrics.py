import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import convolve


def abs2(x):
    return np.array([i**2 for i in x])

def plot_spectrum(prediction, ground_truth, FS, FC_TX, save_dir):
    ax = plt.subplot(1,1,1)
    psd_RX,f = ax.psd(prediction, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10), noverlap = 1, pad_to = 2048)
    psd_NF,f = ax.psd(ground_truth, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10), noverlap = 1, pad_to = 2048)
    ax.set_ylabel(r'PSD, $V^2$/Hz [dB]')
    ax.set_xlabel('Frequency, MHz')
    ax.set_title('Power spectral density')
    plt.show()
    plt.savefig(save_dir+'/img.png')

def cal_power(signal, FS, FC_TX = 0, PIM_SFT = 0, PIM_total_BW = 0):
    ax = plt.subplot(1,1,1)
    plt.close()
    psd_sig,f = ax.psd(signal, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10), 
                          noverlap = 1, pad_to = 2048, label='Residual')

    rng = np.where( (f > FC_TX + PIM_SFT - PIM_total_BW/2) & (f < FC_TX + PIM_SFT + PIM_total_BW/2) )
    psd_temp = psd_sig[rng[0]]
    psd_temp_mean = np.mean(psd_temp.real)
    psd_power = 10*np.log10(psd_temp_mean)
    return psd_power 

def calc_perf(PIM_level,RES_level):
    perf = 10*np.log10(10**((PIM_level)/10) - 1) - 10*np.log10(10**((RES_level)/10) - 1)
    # perf = 10*np.log10(10**((PIM_level + 100)/10) - 1) - 10*np.log10(10**((RES_level + 100)/10) - 1)
    return perf

def calculate_metrics(initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW):

    initial_power = cal_power(initial_signal, FS = FS, FC_TX = FC_TX, PIM_SFT = PIM_SFT, PIM_total_BW = PIM_total_BW)
    filt_power = cal_power(filt_signal, FS, FC_TX = FC_TX, PIM_SFT = PIM_SFT, PIM_total_BW = PIM_total_BW)

    metrics = calc_perf(initial_power, filt_power)
    return metrics

def main_metrics(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_total_BW):

    initial_signal = ground_truth[..., 0].reshape(1, -1)[0] + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    PIM_pred = prediction[..., 0].reshape(1, -1)[0] + 1j * prediction[..., 1].reshape(1, -1)[0]

    filt_signal = initial_signal - PIM_pred

    main_metric = calculate_metrics(initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW)
    # plot_spectrum(initial_signal, filt_signal, FS, FC_TX, PIM_SFT)
    
    return main_metric



def reduction_level(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_BW, noise, filter):

    

    initial_signal = ground_truth[..., 0].reshape(1, -1)[0] + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    PIM_pred = prediction[..., 0].reshape(1, -1)[0] + 1j * prediction[..., 1].reshape(1, -1)[0]

    noise_level = noise[..., 0].reshape(1, -1)[0] + 1j * noise[..., 1].reshape(1, -1)[0]
    filt_conv = filter.astype(complex).flatten()

    min_len = min(noise_level.shape[0], prediction.shape[0])
    
    convolved_initial_signal = convolve(initial_signal, filt_conv)
    residual =  convolve(PIM_pred[:min_len], filt_conv) + convolve(noise_level[:min_len], filt_conv) - convolve(initial_signal[:min_len], filt_conv)

    # TODO: some bug with noise level, need to investigate
    # residual =  convolve(PIM_pred, filt_conv)  - convolve(initial_signal, filt_conv)

    red_level = calculate_metrics(convolved_initial_signal, residual, FS, FC_TX, PIM_SFT, PIM_BW) 
    return red_level












