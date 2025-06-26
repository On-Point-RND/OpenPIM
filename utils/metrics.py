import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import convolve, welch


def count_net_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes
    return n_param


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
    data_type='synth',
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
            data_type,
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
    data_type,
    save_dir,
    path_dir_save="",
    cut=False,
    phase_name="",
):
    # Create new figure with legend
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    _, _ = ax.psd(
            prediction,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label="Predicted Signal",
        )
    _, _ = ax.psd(
            ground_truth,
            Fs=FS,
            Fc=FC_TX,
            NFFT=2048,
            window=np.kaiser(2048, 10),
            noverlap=1,
            pad_to=2048,
            label="Original Signal",
        )
    _, _ = ax.psd(
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
        if data_type == 'synth':
            ax.set_xlim(
                FC_TX - FS / 10 + PIM_SFT - PIM_BW / 2,
                FC_TX + FS / 10 + PIM_SFT + PIM_BW / 2,
            )
        elif data_type == 'real':
            ax.set_xlim( 
                FC_TX  - FS / 10 - 5 / 2 - 8.5,
                FC_TX  + FS / 10 + 5 / 2 + 9.5,
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


def plot_final_spectrums(
    prediction,
    ground_truth,
    noise,
    FS,
    FC_TX,
    PIM_SFT,
    PIM_BW,
    iteration,
    data_type,
    save_dir,
    path_dir_save="",
    phase_name="test",
):

    n_channels = prediction.shape[1]
    dim_1 = int(np.sqrt(n_channels))
    dim_2 = n_channels // dim_1

    if dim_1 * dim_2 > 1:
        fig, axes = plt.subplots(dim_1, dim_2, figsize=(15, 15))
    else: 
        fig, axes = plt.subplots(dim_1, dim_2, figsize=(7, 7))

    for ch_dim_1 in range(dim_1):
        for ch_dim_2 in range(dim_2):

            if dim_1 * dim_2 > 1:
                ax = axes[ch_dim_1][ch_dim_2]
            else: 
                ax = axes

            _, _ = ax.psd(
                    ground_truth[:, ch_dim_1*dim_2 + ch_dim_2],
                    Fs=FS,
                    Fc=FC_TX,
                    NFFT=2048,
                    window=np.kaiser(2048, 10),
                    noverlap=1,
                    pad_to=2048,
                    label="RX",
                    color = 'blue'
                )
            _, _ = ax.psd(
                    ground_truth[:, ch_dim_1*dim_2 + ch_dim_2] - prediction[:, ch_dim_1*4 + ch_dim_2],
                    Fs=FS,
                    Fc=FC_TX,
                    NFFT=2048,
                    window=np.kaiser(2048, 10),
                    noverlap=1,
                    pad_to=2048,
                    label="ERR",
                    color = 'red'
                )
            _, _ = ax.psd(
                    noise[:, ch_dim_1*dim_2 + ch_dim_2],
                    Fs=FS,
                    Fc=FC_TX,
                    NFFT=2048,
                    window=np.kaiser(2048, 10),
                    noverlap=1,
                    pad_to=2048,
                    label="NF",
                    color = 'black'
                )

            # Add plot elements
            ax.set_ylabel(r"PSD, $V^2$/Hz [dB]", fontsize = 16)
            ax.set_xlabel("Frequency, MHz", fontsize = 16)
            ax.set_ylim(0, 48)
            if data_type == 'synth':    
                ax.set_xlim(
                        FC_TX - FS / 10 + PIM_SFT - PIM_BW / 2,
                        FC_TX + FS / 10 + PIM_SFT + PIM_BW / 2,
                    )
            elif data_type == 'real':
                ax.set_xlim( 
                    FC_TX  - FS / 10 - 5 / 2 - 8.5,
                    FC_TX  + FS / 10 + 5 / 2 + 9.5,
                )
            ax.legend(loc="upper left", fontsize = 13)
            ax.set_title(f'CH_{ch_dim_1*4+ch_dim_2}', fontsize = 18)
            ax.grid(True)
    fig.tight_layout()
    fig.show()
    fig.savefig(
        f"{save_dir}/{phase_name}_total_performance_{iteration}_iterations"
        + path_dir_save + ".png",
        # bbox_inches="tight",
    )
    plt.close()


def plot_total_perf(powers, max_red_level, mean_red_level, path_save):
    fig = plt.figure(figsize = (10, 7))
    n_channels = len(powers['gt'])
    nfas = [1] * n_channels
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
    plt.savefig(
        f'{path_save}/' 'barplot_performance.png', bbox_inches='tight'
    )
    plt.close()


def compute_power(
        x, data_type,
        fs, pim_sft, pim_bw,
        real_data_name = '', return_db=True
    ):
    """
    Power calculation using Welch's method without matplotlib
    """
    n = 2048
    # Compute PSD using Scipy's optimized Welch implementation
    f, psd = welch(
        x, fs, window=np.kaiser(2048, 10),
        nperseg=n, noverlap=1, return_onesided=False
    )

    # Calculate frequency mask directly
    if data_type == 'synth':
        freq_mask = np.where(
            (f > pim_sft - pim_bw / 2) & (f < pim_sft + pim_bw / 2)
        )
    elif data_type == 'real':
        if real_data_name == 'data_A':
            freq_mask = np.where((f >  - 5 / 2 - 27.5) & (f < 5 / 2 - 27.5))
        elif real_data_name == 'set_B':
            freq_mask = np.where((f >  - 5 / 2 + 32.5) & (f < 5 / 2 + 32.5))
        else:
            freq_mask = np.where((f >  - 5 / 2 + 15) & (f < 5 / 2 + 15))

    psd_window = psd[freq_mask[0]]

    power = np.mean(psd_window.real)
    if return_db:
        power = 10 * np.log10(power)
    return power


def calc_perf(orig_pwr, residual_pwr):
    perf = 10 * np.log10(10 ** (orig_pwr / 10) - 1) - 10 * np.log10(
        10 ** (residual_pwr / 10) - 1
    )
    return perf


def calculate_res(
        orig_signal, residual_signal, data_type,
        fs, pim_sft, pim_bw, real_data_name
    ):
    orig_power = compute_power(
        orig_signal, data_type,
        fs, pim_sft, pim_bw,
        real_data_name
    )
    residual_power = compute_power(
        residual_signal, data_type,
        fs, pim_sft, pim_bw,
        real_data_name
    )
    metrics = calc_perf(orig_power, residual_power)
    return metrics


def reduction_level(
        prediction, ground_truth, data_type,
        fs, pim_sft, pim_bw,
        filter, real_data_name
    ):

    orig_signal = (
        ground_truth[..., 0].reshape(1, -1)[0]
        + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    )
    pred_signal = (
        prediction[..., 0].reshape(1, -1)[0]
        + 1j * prediction[..., 1].reshape(1, -1)[0]
    )

    filt_conv = filter.astype(complex).flatten()
    convolved_orig_signal = convolve(orig_signal, filt_conv)
    convolved_pred_signal = convolve(pred_signal, filt_conv)
    residual = convolved_pred_signal - convolved_orig_signal

    red_level = calculate_res(
        convolved_orig_signal, residual, data_type,
        fs, pim_sft, pim_bw, real_data_name
    )
    return red_level
