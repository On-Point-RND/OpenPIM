import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(data):
    
    rxa = data["rxa"][0]
    txa = data["txa"][0]
    nfa = data["nfa"][0]
    
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FC_RX = data['BANDS_UL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6 
    PIM_SFT = data['PIM_sft'][0][0] / 10**6
    PIM_BW = data['BANDS_TX'][0][0][1][0][0] / 10**6
    
    ax = plt.subplot(1,1,1)
    psd_RX,f = ax.psd(rxa, Fs = FS, Fc = FC_TX,
                      NFFT = 2048, window = np.kaiser(2048,10),
                      noverlap = 1, pad_to = 2048)
    psd_NF,f = ax.psd(nfa, Fs = FS, Fc = FC_TX,
                      NFFT = 2048, window = np.kaiser(2048,10),
                      noverlap = 1, pad_to = 2048)
    ax.set_ylabel(r'PSD, $V^2$/Hz [dB]')
    ax.set_xlabel('Frequency, MHz')
    ax.set_title('Power spectral density')
    plt.show()
