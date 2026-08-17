import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple
from scipy.signal import convolve, welch

from modules.data_utils import toComplex


_PSD_KWARGS = {
    "NFFT": 2048,
    "window": np.kaiser(2048, 10),
    "noverlap": 1,
    "pad_to": 2048,
}

_PSD_NFFT = 2048


def compute_psd(x: np.ndarray, fs: float, nfft: int = _PSD_NFFT):
    """
    Two-sided Welch PSD (linear). Matches classic metric window settings.
    Returns freqs (Hz, baseband), psd (linear power density).
    """
    freqs, psd = welch(
        x,
        fs,
        window=np.kaiser(nfft, 10),
        nperseg=nfft,
        noverlap=1,
        return_onesided=False,
    )
    return freqs, np.maximum(np.asarray(psd.real), 1e-30)


def compute_spectra_bundle(
    gt_rescaled,
    pred_rescaled,
    noise,
    fs: float,
):
    """
    Per-channel PSDs for rx / pred / err / noise.
    Inputs are real I/Q arrays (T, C, 2). Returns freqs and arrays (C, F).
    """
    gt = toComplex(gt_rescaled)
    pred = toComplex(pred_rescaled)
    nse = toComplex(noise)
    err = gt - pred
    n_channels = gt.shape[1]

    freqs = None
    stacks = {"rx": [], "pred": [], "err": [], "noise": []}
    for ch in range(n_channels):
        for key, sig in (
            ("rx", gt[:, ch]),
            ("pred", pred[:, ch]),
            ("err", err[:, ch]),
            ("noise", nse[:, ch]),
        ):
            f, p = compute_psd(sig, fs)
            if freqs is None:
                freqs = f
            stacks[key].append(p)

    return freqs, {k: np.stack(v, axis=0) for k, v in stacks.items()}


def _set_pim_xlim(ax, data_type, FC_TX, FS, PIM_SFT, PIM_BW):
    if data_type == "synth":
        ax.set_xlim(
            FC_TX - FS / 10 + PIM_SFT - PIM_BW / 2,
            FC_TX + FS / 10 + PIM_SFT + PIM_BW / 2,
        )
    elif data_type == "real":
        ax.set_xlim(
            FC_TX - FS / 10 - 5 / 2 - 8.5,
            FC_TX + FS / 10 + 5 / 2 + 9.5,
        )


def count_net_params(net):
    n_param = 0
    for _, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes
    return n_param


def NMSE(prediction, ground_truth):
    i_hat = prediction[..., 0]
    i_true = ground_truth[..., 0]
    q_hat = prediction[..., 1]
    q_true = ground_truth[..., 1]

    MSE = np.mean(
        np.square(i_true - i_hat) + np.square(q_true - q_hat),
        axis=-1,
    )
    energy = np.mean(
        np.square(i_true) + np.square(q_true),
        axis=-1,
    )

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
            data_type,
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
    data_type,
    save_dir,
    path_dir_save="",
    cut=False,
    phase_name="",
):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for signal, label in (
        (prediction, "Predicted Signal"),
        (ground_truth, "Original Signal"),
        (ground_truth - prediction, "(Original - Predicted) Signal"),
    ):
        ax.psd(
            signal,
            Fs=FS,
            Fc=FC_TX,
            label=label,
            **_PSD_KWARGS,
        )

    ax.set_ylabel(r"PSD, $V^2$/Hz [dB]")
    ax.set_xlabel("Frequency, MHz")
    if cut:
        _set_pim_xlim(ax, data_type, FC_TX, FS, PIM_SFT, PIM_BW)

    ax.set_title(
        f"{phase_name} Power Spectral Density - Iteration: {iteration}, "
        f"Reduction: {reduction_level:.3f} dB, "
        f"CH_{c_number}"
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
    ncols = int(np.ceil(np.sqrt(n_channels)))
    nrows = int(np.ceil(n_channels / ncols))
    figsize = (7 * ncols, 5 * nrows) if n_channels <= 4 else (15, 15)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ch_idx in range(n_channels):
        ax = axes_flat[ch_idx]
        for signal, label, color in (
            (ground_truth[:, ch_idx], "RX", "blue"),
            (ground_truth[:, ch_idx] - prediction[:, ch_idx], "ERR", "red"),
            (noise[:, ch_idx], "NF", "black"),
        ):
            ax.psd(
                signal,
                Fs=FS,
                Fc=FC_TX,
                label=label,
                color=color,
                **_PSD_KWARGS,
            )

        ax.set_ylabel(r"PSD, $V^2$/Hz [dB]", fontsize=16)
        ax.set_xlabel("Frequency, MHz", fontsize=16)
        ax.set_ylim(0, 48)
        _set_pim_xlim(ax, data_type, FC_TX, FS, PIM_SFT, PIM_BW)
        ax.legend(loc="upper left", fontsize=13)
        ax.set_title(f"CH_{ch_idx}", fontsize=18)
        ax.grid(True)

    for ax in axes_flat[n_channels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        f"{save_dir}/{phase_name}_total_performance_{iteration}_iterations"
        + path_dir_save
        + ".png",
    )
    plt.close(fig)

def compute_powers_dict(
    gt_rescaled,
    pred_rescaled,
    noise_test,
    signal_specs: Tuple[float, float, float, str, str],
) -> dict:
    """Build dict of per-channel powers for gt, err, noise.
    signal_specs: (FS, PIM_SFT, PIM_BW, data_type, data_name).
    """
    FS, PIM_SFT, PIM_BW, data_type, data_name = signal_specs
    powers = {}
    for key, value in (
        ("gt", gt_rescaled),
        ("err", gt_rescaled - pred_rescaled),
        ("noise", noise_test),
    ):
        compl = toComplex(value)
        powers[key] = [
            compute_power(
                compl[:, ch_id],
                FS, PIM_SFT, PIM_BW,
                data_type, data_name,
            )
            for ch_id in range(compl.shape[1])
        ]
    return powers


def perf_from_powers(powers: dict) -> float:
    """Mean performance (reduction level) from powers dict with keys gt, err, noise."""
    n_channels = len(powers["gt"])
    gt_norm = [powers["gt"][i] - powers["noise"][i] for i in range(n_channels)]
    err_norm = [powers["err"][i] - powers["noise"][i] for i in range(n_channels)]
    perf_list = [calc_perf(gt_norm[i], err_norm[i]) for i in range(n_channels)]
    return perf_list


def plot_total_perf(powers, path_save):
    _ = plt.figure(figsize=(10, 7))
    n_channels = len(powers["gt"])
    gt_norm = [powers["gt"][idx] - powers["noise"][idx] for idx in range(n_channels)]
    err_norm = [powers["err"][idx] - powers["noise"][idx] for idx in range(n_channels)]

    power_df = pd.DataFrame({"RXA": gt_norm, "ERR": err_norm})

    perf_list = perf_from_powers(powers)
    mean_perf = calculate_mean_red(perf_list)
    max_perf = max(perf_list)

    power_df.plot.bar(color=("red", "blue", "black"))
    plt.title(
        f"PIM: "
        f"ORIG: {calculate_mean_red(power_df['RXA']):.2f}, "
        f"RES: {calculate_mean_red(power_df['ERR']):.2f}; "
        f"Perf. ABS: {max_perf:.2f}, "
        f"MEAN: {mean_perf:.2f}"
    )
    plt.xlabel("Channel number", fontsize=16)
    plt.ylabel("Signal level [dB]", fontsize=16)
    plt.legend(loc="upper left")
    plt.savefig(f"{path_save}/barplot_performance.png", bbox_inches="tight")
    plt.close()


def compute_power_lite(
    x,
    fs,
    pim_sft,
    pim_bw,
    data_type,
    real_data_name="",
    return_db=True,
):
    """
    Band power via Welch: Hann window, K_FFT=2048, 50% overlap (noverlap=nperseg//2).
    Mean over linear PSD bins in the receive band, then optionally to dB.
    """
    n = 2048
    f, psd = welch(
        x,
        fs,
        window="hann",
        nperseg=n,
        noverlap=n // 2,
        return_onesided=False,
    )

    if data_type == "synth":
        freq_mask = np.where(
            (f > pim_sft - pim_bw / 2) & (f < pim_sft + pim_bw / 2)
        )
    elif data_type == "real":
        if real_data_name == "data_A":
            freq_mask = np.where((f > -5 / 2 - 27.5) & (f < 5 / 2 - 27.5))
        elif real_data_name == "set_B":
            freq_mask = np.where((f > -5 / 2 + 32.5) & (f < 5 / 2 + 32.5))
        else:
            freq_mask = np.where((f > -5 / 2 + 15) & (f < 5 / 2 + 15))

    power = np.mean(psd[freq_mask[0]].real)
    if return_db:
        power = 10 * np.log10(power)
    return power


def compute_powers_dict_lite(
    gt_rescaled,
    pred_rescaled,
    signal_specs: Tuple[float, float, float, str, str],
) -> dict:
    """
    Per-channel RXA (gt) and RES (gt-pred) levels [dB] via compute_power_lite.
    Same layout as compute_powers_dict: pred/gt already band-filtered upstream;
    no extra FIR, no noise. signal_specs: (FS, PIM_SFT, PIM_BW, data_type, data_name).
    """
    FS, PIM_SFT, PIM_BW, data_type, data_name = signal_specs
    powers = {}
    for key, value in (
        ("gt", gt_rescaled),
        ("err", gt_rescaled - pred_rescaled),
    ):
        compl = toComplex(value)
        powers[key] = [
            compute_power_lite(
                compl[:, ch_id],
                FS,
                PIM_SFT,
                PIM_BW,
                data_type,
                data_name,
            )
            for ch_id in range(compl.shape[1])
        ]
    return powers


def perf_from_powers_lite(powers: dict) -> float:
    """Mean residual level D: arithmetic average of per-antenna D_n [dB]."""
    return float(np.mean(powers["err"]))


def plot_total_perf_lite(powers, path_save):
    """Final barplot RXA vs RES; saves barplot_performance_lite.png."""
    _ = plt.figure(figsize=(10, 7))
    gt = powers["gt"]
    err = powers["err"]
    mean_rxa = float(np.mean(gt))
    mean_res = perf_from_powers_lite(powers)

    power_df = pd.DataFrame({"RXA": gt, "RES": err})
    power_df.plot.bar(color=("tab:red", "tab:blue"))
    plt.title(
        f"Residual level D (lite): "
        f"RXA = {mean_rxa:.2f} dB, RES = {mean_res:.2f} dB"
    )
    plt.xlabel("Channel number", fontsize=16)
    plt.ylabel("Signal level [dB]", fontsize=16)
    plt.legend(loc="upper left")
    plt.savefig(f"{path_save}/barplot_performance_lite.png", bbox_inches="tight")
    plt.close()


def compute_power(
        x,
        fs,
        pim_sft,
        pim_bw,
        data_type,
        real_data_name = '',
        return_db=True
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


def reduction_level(
        prediction, ground_truth, data_type,
        fs, pim_sft, pim_bw,
        filter, real_data_name, with_noise = True, noise = None
    ):
    filt_conv = filter.astype(complex).flatten()

    orig_signal = (
        ground_truth[..., 0].reshape(1, -1)[0]
        + 1j * ground_truth[..., 1].reshape(1, -1)[0]
    )
    pred_signal = (
        prediction[..., 0].reshape(1, -1)[0]
        + 1j * prediction[..., 1].reshape(1, -1)[0]
    )
    if with_noise:
        assert noise is not None
        noised_signal = (
            noise[..., 0].reshape(1, -1)[0]
            + 1j * noise[..., 1].reshape(1, -1)[0]
        )
        convolved_noise = convolve(noised_signal, filt_conv)

    convolved_orig_signal = convolve(orig_signal, filt_conv)
    convolved_pred_signal = convolve(pred_signal, filt_conv)
    residual = convolved_pred_signal - convolved_orig_signal

    orig_power = compute_power(
        convolved_orig_signal,
        fs, pim_sft, pim_bw,
        data_type,
        real_data_name
    )
    residual_power = compute_power(
        residual,
        fs, pim_sft, pim_bw,
        data_type,
        real_data_name
    )
    if with_noise:
        noise_power = compute_power(
        convolved_noise,
        fs, pim_sft, pim_bw,
        data_type,
        real_data_name
    )
        orig_power = orig_power - noise_power
        residual_power = residual_power - noise_power

    red_level = calc_perf(orig_power, residual_power)
    return red_level


def calculate_mean_red(red_levels: Iterable[float]) -> float:
    power_levels = []
    for red_level in red_levels:
        power_levels.append(10 ** (red_level / 10))
    return 10 * np.log10(np.mean(power_levels))


def calculate_metrics(
    prediction, ground_truth, noise,
    filter, data_type, data_name,
    СScaler, fs, pim_sft, pim_bw, stat
):
    if not "NMSE" in stat:
        stat["NMSE"] = dict()

    if not "Reduction_level" in stat:
        stat["Reduction_level"] = dict()

    n_channels = prediction.shape[1]

    pred = СScaler.rescale(prediction, key="Y")
    gt = СScaler.rescale(ground_truth, key="Y")

    for c in range(n_channels):
        stat["NMSE"][f"CH_{c}"] = NMSE(prediction, ground_truth)
        stat["Reduction_level"][f"CH_{c}"] = reduction_level(
            pred[:, c],
            gt[:, c],
            data_type,
            fs,
            pim_sft,
            pim_bw,
            filter,
            data_name,
            noise = noise
        )
    return stat
