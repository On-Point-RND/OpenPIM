import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import convolve, welch


def abs2(x):
    return np.array([i**2 for i in x])


def plot_spectrum(prediction, ground_truth, FS, FC_TX, save_dir):
    ax = plt.subplot(1,1,1)
    psd_RX,f = ax.psd(prediction, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10), noverlap = 1, pad_to = 2048)
    psd_NF,f = ax.psd(ground_truth, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10), noverlap = 1, pad_to = 2048)
    ax.set_ylabel(r'PSD, $V^2$/Hz [dB]')
    ax.set_xlabel('Frequency, MHz')
    ax.set_title('Power spectral density')
    #plt.show()
    plt.savefig(save_dir+'/img.png')


def compute_power(x, fs, pim_sft, pim_bw, return_db=True):
    """
    Power calculation using Welch's method without matplotlib
    """
    n = 2048
    # Compute PSD using Scipy's optimized Welch implementation
    f, psd = welch(
        x, fs, window=np.kaiser(2048,10), nperseg=n, noverlap=1
    )
    # Calculate frequency mask directly
    freq_mask = np.where(
        (f > pim_sft - pim_bw/2) & (f < pim_sft + pim_bw/2)
    )
    psd_window = psd[freq_mask[0]]
    power = np.mean(psd_window.real)
    if return_db:
        power = 10 * np.log10(power)
    return power


def calc_perf(PIM_level,RES_level):
    perf = 10*np.log10(10**((PIM_level)/10) - 1) - 10*np.log10(10**((RES_level)/10) - 1)
    # perf = 10*np.log10(10**((PIM_level + 100)/10) - 1) - 10*np.log10(10**((RES_level + 100)/10) - 1)
    return perf


def calculate_metrics(initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW):
    initial_power = compute_power(initial_signal, FS, PIM_SFT, PIM_total_BW)
    filt_power = compute_power(filt_signal, FS, PIM_SFT, PIM_total_BW)
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

    # # TODO: some bug with noise level, need to investigate
    # residual =  convolve(PIM_pred, filt_conv) + convolve(noise_level, filt_conv) - convolve(initial_signal, filt_conv)

    red_level = calculate_metrics(convolved_initial_signal, residual, FS, FC_TX, PIM_SFT, PIM_BW) 
    return red_level
