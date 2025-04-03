import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import convolve, welch


def abs2(x):
    return np.array([i**2 for i in x])

import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_BW, iteration, reduction_level, n_channel, save_dir, add_name='',
                  cut=False, initial=False, initial_ground_truth=[]):
    """
    Plots the power spectral density (PSD) of signals and saves the resulting figure.

    Parameters:
        prediction: Predicted signal data.
        ground_truth: Ground truth signal data.
        FS: Sampling frequency.
        FC_TX: Center frequency.
        PIM_SFT: Frequency shift for PIM.
        PIM_BW: Bandwidth for PIM.
        iteration: Current iteration number.
        reduction_level: Reduction level in dB.
        n_channel: Channel number.
        save_dir: Directory to save the plot.
        add_name: Optional suffix for the saved file name.
        cut: Whether to limit the x-axis range.
        initial: Whether to include the initial ground truth signal.
        initial_ground_truth: Initial ground truth signal data (used if `initial=True`).
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot the predicted signal
    psd_RX, f = ax.psd(
        prediction,
        Fs=FS,
        Fc=FC_TX,
        NFFT=2048,
        window=np.kaiser(2048, 10),
        noverlap=1,
        pad_to=2048,
        label='Predicted Signal'
    )

    # Plot the ground truth or initial signals based on the `initial` flag
    if initial:
        # Plot the initial ground truth signal
        psd_NF, f = ax.psd(
            initial_ground_truth,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label='Initial Signal'
        )
        # Plot the residual signal
        psd_NF, f = ax.psd(
            ground_truth,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label='Residual Signal'
        )
        # Plot the difference between residual and predicted signals
        psd_NF, f = ax.psd(
            ground_truth - prediction,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label='(Residual - Predicted) Signal'
        )
    else:
        # Plot the original signal
        psd_NF, f = ax.psd(
            ground_truth,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label='Original Signal'
        )
        # Plot the difference between original and predicted signals
        psd_NF, f = ax.psd(
            ground_truth - prediction,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label='(Original - Predicted) Signal'
        )

    # Add plot elements
    ax.set_ylabel(r'PSD, $V^2$/Hz [dB]')
    ax.set_xlabel('Frequency, MHz')
    if cut:
        ax.set_xlim(FC_TX - FS / 10 + PIM_SFT - PIM_BW / 2, FC_TX + FS / 10 + PIM_SFT + PIM_BW / 2)
    ax.set_title(f'Power Spectral Density - Iteration: {iteration}, Reduction: {reduction_level:.3f} dB, CH_{n_channel}')

    # Place the legend in the upper left corner
    ax.legend(loc='upper left')

    # Save the figure
    if cut:
        plt.savefig(f'{save_dir}/img_{iteration}_cut{add_name}.png', bbox_inches='tight')
    else:
        plt.savefig(f'{save_dir}/img_{iteration}{add_name}.png', bbox_inches='tight')

    # Close the figure to prevent accumulation
    plt.close()
    
def compute_power(x, fs, fc_tx, pim_sft, pim_bw, return_db=True):
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


def calculate_res(initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW):
    initial_power = compute_power(initial_signal, FS, FC_TX, PIM_SFT, PIM_total_BW)
    filt_power = compute_power(filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW)
    metrics = calc_perf(initial_power, filt_power)
    return metrics


def calculate_avg_metrics(orig_signal: np.ndarray, filt_signal: np.ndarray,
                          fs, pim_sft, pim_bw):
    """
    Computes average metrics across n transceivers.
    Requires signals in a shape (k x n),
    where k is a length of a signal sample
    """
    assert len(orig_signal.shape) > 1
    assert len(filt_signal.shape) > 1
    n_trans = orig_signal.shape[1]
    metrics = 0.0
    for i in range(n_trans):
        init_power = compute_power(orig_signal[:, i], fs, pim_sft, pim_bw)
        filt_power = compute_power(filt_signal[:, i], fs, pim_sft, pim_bw)
        metrics += calc_perf(init_power, filt_power)
    return (metrics / n_trans)


def main_metrics(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_total_BW):
    initial_signal = ground_truth[..., 0].reshape(1, -1)[0] + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    PIM_pred = prediction[..., 0].reshape(1, -1)[0] + 1j * prediction[..., 1].reshape(1, -1)[0]

    filt_signal = initial_signal - PIM_pred

    main_metric = calculate_res(initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW)
    # plot_spectrum(initial_signal, filt_signal, FS, FC_TX, PIM_SFT)
    
    return main_metric


def reduction_level(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_BW, noise, filter):    

    initial_signal = ground_truth[..., 0].reshape(1, -1)[0] + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    PIM_pred = prediction[..., 0].reshape(1, -1)[0] + 1j * prediction[..., 1].reshape(1, -1)[0]

    noise_level = noise[..., 0].reshape(1, -1)[0] + 1j * noise[..., 1].reshape(1, -1)[0]

    filt_conv = filter.astype(complex).flatten()

    # min_len = min(noise_level.shape[0], prediction.shape[0])
    
    convolved_initial_signal = convolve(initial_signal, filt_conv)
    # residual =  convolve(PIM_pred[:min_len], filt_conv) - convolve(initial_signal[:min_len], filt_conv)
    residual =  convolve(PIM_pred, filt_conv) - convolve(initial_signal, filt_conv)

    # # TODO: some bug with noise level, need to investigate
    # residual =  convolve(PIM_pred, filt_conv) + convolve(noise_level, filt_conv) - convolve(initial_signal, filt_conv)

    red_level = calculate_res(convolved_initial_signal, residual, FS, FC_TX, PIM_SFT, PIM_BW) 
    return red_level