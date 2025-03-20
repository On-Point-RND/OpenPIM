from typing import Tuple
import numpy as np


def magnitude_spectrum(input_signal: np.ndarray[np.complex128],
                       sample_rate: int,
                       nfft: int,
                       shift: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform (FFT) of the input signal.

    Parameters:
    - input_signal (np.ndarray[np.complex128]): A 2D numpy array where the first dimension
                                               represents batch size and the second dimension
                                               represents the time sequence of complex numbers.
    - sample_rate (int): The rate at which the input signal was sampled.
    - shift (bool, optional): Whether or not to shift the zero-frequency component to
                              the center of the spectrum. Defaults to False.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple where the first element is the frequency components
                                     and the second element is the FFT of the input signal for each batch.
    """

    # Compute the FFT of the input signal along the last axis (time sequence dimension)
    spectrum = np.fft.fft(input_signal, n=nfft, axis=-1)

    # Shift the zero-frequency component to the center if `shift` is True
    if shift:
        spectrum = np.fft.fftshift(spectrum, axes=-1)
        freq = np.fft.fftshift(np.fft.fftfreq(input_signal.shape[1], d=1 / sample_rate))
    else:
        # Generate the frequencies for the unshifted spectrum
        freq = np.linspace(0, sample_rate, input_signal.shape[1])

    return freq, spectrum


def power_spectrum(complex_signal, fs=800e6, nperseg=2560, axis=-1):
    """
    Compute the Power Spectral Density (PSD) of a given complex signal using the Welch method.

    Parameters:
    - complex_signal: Input complex signal for which the PSD is to be computed.
    - fs (float, optional): Sampling frequency of the signal. Default is 800e6 (800 MHz).
    - nperseg (int, optional): Number of datasets points to be used in each block for the Welch method. Default is 2560.

    Returns:
    - frequencies_signal_subset: Frequencies at which the PSD is computed.
    - psd_signal_subset: PSD values.
    """

    import numpy as np
    from scipy.signal import welch

    # Compute the PSD using the Welch method
    freq, ps = welch(complex_signal, fs=fs, nperseg=nperseg,
                      return_onesided=False, scaling='spectrum', axis=-1)

    # To make the frequency axis monotonic, we need to shift the zero frequency component to the center.
    # This step rearranges the computed PSD and frequency values such that the negative frequencies appear first.
    half_nfft = int(nperseg / 2)
    freq = np.concatenate(
        (freq[half_nfft:], freq[:half_nfft]))

    # Rearrange the PSD values corresponding to the rearranged frequency values.
    ps = np.concatenate((ps[..., half_nfft:], ps[..., :half_nfft]), axis=-1)

    # Take the average of all signals
    ps = np.mean(ps, axis=0)

    return freq, ps


def NMSE(prediction, ground_truth):
    i_hat = prediction[..., 0]
    i_true = ground_truth[..., 0]
    q_hat = prediction[..., 1]
    q_true = ground_truth[..., 1]

    MSE = np.mean(np.square(i_true - i_hat) + np.square(q_true - q_hat), axis=-1)
    energy = np.mean(np.square(i_true) + np.square(q_true), axis=-1)

    NMSE = np.mean(10 * np.log10(MSE / energy))
    return NMSE


def IQ_to_complex(IQ_signal):
    """
    Convert a multi-dimensional array of I-Q pairs into a 2D array of complex signals.

    Args:
    - IQ_in_segment (3D array): The prediction I-Q datasets with shape (#segments, frame_length, 2).

    Returns:
    - 2D array of shape (#segments, frame_length) containing complex signals.
    """

    # Extract I and Q values
    I_values = IQ_signal[..., 0]
    Q_values = IQ_signal[..., 1]

    # Convert to complex signals
    complex_signals = I_values + 1j * Q_values

    return complex_signals
