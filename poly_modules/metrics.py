import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def compute_power(x, fs, pim_sft, pim_bw, return_db=False):
    n = 2048
    f, pxx = welch(
        x, fs, window=np.kaiser(2048,10), nperseg=n, noverlap=1, nfft=n
    )
    rng = np.where(
        (f > pim_sft - pim_bw/2) & (f < pim_sft + pim_bw/2)
    )
    pxx_window = pxx[rng[0]]
    power = np.mean(pxx_window.real)
    if return_db:
        power = 10 * np.log10(power)
    return power


def compute_perf(x_pwr, y_pwr):
    return 10 * np.log10((x_pwr - 1) / (y_pwr - 1))


def compute_db_power(signal, FS, FC_TX = 0, PIM_SFT = 0, PIM_BW = 0):
    """
        Calculate mean signal power in a certain frequency window:
        (FC_TX + PIM_SFT - PIM_BW/2; FC_TX + PIM_SFT + PIM_BW/2)

        Args:
            signal: signal values in time
            FS: sampling frequency (samples per time unit)
            FC_TX: center frequency
            PIM_SFT: PIM shift
            PIM_BW: PIM bandwidth

        Returns:
            psd_power: calculated power in dB

    """
    ax = plt.subplot(1,1,1)
    plt.close()
    psd_sig,f = ax.psd(signal, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10),
                          noverlap = 1, pad_to = 2048, label='Residual')

    rng = np.where( (f > FC_TX + PIM_SFT - PIM_BW/2) & (f < FC_TX + PIM_SFT + PIM_BW/2) )
    psd_temp = psd_sig[rng[0]]
    psd_temp_mean = np.mean(psd_temp.real)
    psd_power = 10*np.log10(psd_temp_mean)
    return psd_power


def compute_perf_db_input(x_pwr_db, y_pwr_db):
    perf = 10 * np.log10(10**(x_pwr_db/10) - 1) - 10 * np.log10(10**(y_pwr_db/10) - 1)
    return perf


def calculate_metrics(initial_signal, filt_signal, FS, PIM_SFT, PIM_BW):
    """
        Calculate metrics for filtered signal compared to initial signal.

        Args:
            initial_signal: initial signal without noise filtering
            filt_signal: signal after noise filtering
            FS: sampling frequency (samples per time unit)
            PIM_SFT: PIM shift
            PIM_BW: PIM bandwidth

        Returns:
            metrics: calculated metric for filtered signal compared to initial signal

    """
    initial_power = compute_power(initial_signal, FS, PIM_SFT, PIM_BW)
    filt_power = compute_power(filt_signal, FS, PIM_SFT, PIM_BW)
    metrics = compute_perf(initial_power, filt_power)
    return metrics


def calculate_avg_metrics(orig_signal, filt_signal, FS, PIM_SFT, PIM_BW):
    assert len(orig_signal.shape) > 1
    assert len(filt_signal.shape) > 1
    n_trans = orig_signal.shape[1]
    metrics = 0.0
    for i in range(n_trans):
        init_power = compute_power(orig_signal[:, i], FS, PIM_SFT, PIM_BW)
        filt_power = compute_power(filt_signal[:, i], FS, PIM_SFT, PIM_BW)
        metrics += compute_perf(init_power, filt_power)
    return (metrics / n_trans)
