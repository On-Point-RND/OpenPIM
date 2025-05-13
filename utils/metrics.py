import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import convolve, welch


def abs2(x):
    return np.array([i**2 for i in x])


def NMSE(prediction, ground_truth):
    i_hat = prediction[..., 0]
    i_true = ground_truth[..., 0]
    q_hat = prediction[..., 1]
    q_true = ground_truth[..., 1]

    MSE = np.mean(np.square(i_true - i_hat) + np.square(q_true - q_hat), axis=-1)
    energy = np.mean(np.square(i_true) + np.square(q_true), axis=-1)

    NMSE = np.mean(10 * np.log10(MSE / energy))
    return NMSE


def plot_spectrums(
    prediction,
    ground_truth,
    FS,
    FC_TX,
    PIM_SFT,
    PIM_BW,
    iteration,
    reduction_level,
    save_dir,
    path_dir_save="",
    cut=False,
    phase_name="test",
):

    n_channels = prediction.shape[1]

    for c_number in range(n_channels):
        plot_spectrum(
            prediction[:, c_number],
            ground_truth[:, c_number],
            FS,
            FC_TX,
            PIM_SFT,
            PIM_BW,
            iteration,
            reduction_level[f"CH_{c_number}"],
            c_number,
            save_dir,
            path_dir_save,
            cut,
            phase_name,
        )


def plot_spectrum(
    prediction,
    ground_truth,
    FS,
    FC_TX,
    PIM_SFT,
    PIM_BW,
    iteration,
    reduction_level,
    c_number,
    save_dir,
    path_dir_save="",
    cut=False,
    phase_name="",
):
    # Create new figure with legend
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    psd_RX, f = ax.psd(
            prediction,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label="Predicted Signal",
        )
    psd_NF, f = ax.psd(
            ground_truth,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label="Original Signal",
        )
    psd_NF, f = ax.psd(
            ground_truth - prediction,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label="(Original - Predicted) Signal",
        )

    # Add plot elements
    ax.set_ylabel(r"PSD, $V^2$/Hz [dB]")
    ax.set_xlabel("Frequency, MHz")
    if cut:
        ax.set_xlim(
            FC_TX - FS / 10 + PIM_SFT - PIM_BW / 2,
            FC_TX + FS / 10 + PIM_SFT + PIM_BW / 2,
        )
    ax.set_title(
        f"{phase_name} Power Spectral Density - Iteration: {iteration}, Reduction: {reduction_level:.3f} dB, CH_{c_number}"
    )
    ax.legend(loc="upper left")

    if cut:
        plt.savefig(
            f"{save_dir}/img_{phase_name}_{iteration}_cut_CH{c_number}"
            + path_dir_save
            + ".png",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{save_dir}/img_{phase_name}_{iteration}_CH{c_number}"
            + path_dir_save
            + ".png",
            bbox_inches="tight",
        )
    plt.close()  # Prevent figure accumulation


def plot_total_perf(powers, max_red_level, mean_red_level, path_save):
    fig = plt.figure(figsize = (10, 7))

    n_channels = len(powers['gt'])
    nfas = [1]*n_channels
    gt_norm = [powers['gt'][idx] - powers['noise'][idx] + 1 for idx in range(n_channels)]
    err_norm = [powers['err'][idx] - powers['noise'][idx] + 1 for idx in range(n_channels)]

    power_df = pd.DataFrame({
    'RXA':gt_norm,
    'ERR':err_norm,
    'NFA':nfas
    })

    power_df.plot.bar(color = ('red', 'blue', 'black'))
    plt.title(
        f'PIM: '
        f'ORIG: {round(np.mean(power_df["RXA"]) - 1, 2)}, '
        f'RES: {round(np.mean(power_df["ERR"]) - 1, 2)}; '
        f'Performance ABS: {round(max_red_level, 2)}, '
        f'MEAN: {round(mean_red_level, 2)}'
    )
    plt.xlabel('Channel number', fontsize = 16)
    plt.ylabel('Signal level [dB]', fontsize = 16)
    plt.legend(loc="upper left")
    plt.savefig(f'{path_save}/' 'barplot_perfofmance.png', bbox_inches='tight')
    plt.close()


def compute_power(x, fs, fc_tx, pim_sft, pim_bw, return_db=True):
    """
    Power calculation using Welch's method without matplotlib
    """
    n = 2048
    # Compute PSD using Scipy's optimized Welch implementation
    f, psd = welch(
        x, fs, window=np.kaiser(2048, 10), nperseg=n, noverlap=1, return_onesided=False
    )
    # Calculate frequency mask directly
    freq_mask = np.where((f > pim_sft - pim_bw / 2) & (f < pim_sft + pim_bw / 2))

    psd_window = psd[freq_mask[0]]
    power = np.mean(psd_window.real)
    if return_db:
        power = 10 * np.log10(power)
    return power


def calc_perf(PIM_level, RES_level):
    perf = 10 * np.log10(10 ** ((PIM_level) / 10) - 1) - 10 * np.log10(
        10 ** ((RES_level) / 10) - 1
    )
    return perf


def calculate_res(initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW):
    initial_power = compute_power(initial_signal, FS, FC_TX, PIM_SFT, PIM_total_BW)
    filt_power = compute_power(filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW)
    metrics = calc_perf(initial_power, filt_power)
    return metrics


def calculate_avg_metrics(
    orig_signal: np.ndarray, filt_signal: np.ndarray, fs, pim_sft, pim_bw
):
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
    return metrics / n_trans


def main_metrics(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_total_BW):
    initial_signal = (
        ground_truth[..., 0].reshape(1, -1)[0]
        + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    )
    PIM_pred = (
        prediction[..., 0].reshape(1, -1)[0] + 1j * prediction[..., 1].reshape(1, -1)[0]
    )

    filt_signal = initial_signal - PIM_pred

    main_metric = calculate_res(
        initial_signal, filt_signal, FS, FC_TX, PIM_SFT, PIM_total_BW
    )
    return main_metric


def reduction_level(prediction, ground_truth, FS, FC_TX, PIM_SFT, PIM_BW, filter):

    initial_signal = (
        ground_truth[..., 0].reshape(1, -1)[0]
        + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    )
    PIM_pred = (
        prediction[..., 0].reshape(1, -1)[0] + 1j * prediction[..., 1].reshape(1, -1)[0]
    )

    filt_conv = filter.astype(complex).flatten()

    convolved_initial_signal = convolve(initial_signal, filt_conv)

    residual = convolve(PIM_pred, filt_conv) - convolve(initial_signal, filt_conv)

    red_level = calculate_res(
        convolved_initial_signal, residual, FS, FC_TX, PIM_SFT, PIM_BW
    )
    return red_level
