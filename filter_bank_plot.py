import numpy as np
from spafe.utils.converters import erb2hz
from spafe.utils.vis import show_fbanks
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks

if __name__ == '__main__':
    # init var
    fs = 4000
    nfilts = 16
    nfft = 512
    low_freq = 0
    high_freq = fs / 2

    # compute freqs for xaxis
    ghz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)
    scale = "constant"
    # gamma fbanks
    gamma_fbanks_mat, gamma_freqs = gammatone_filter_banks(nfilts=nfilts,
                                                           nfft=nfft,
                                                           fs=fs,
                                                           low_freq=low_freq,
                                                           high_freq=high_freq,
                                                           scale=scale,
                                                           order=4)
    # visualize filter bank
    show_fbanks(
        gamma_fbanks_mat,
        [erb2hz(freq) for freq in gamma_freqs],
        ghz_freqs,
        ylabel="归一化幅值",
        x1label="频率/Hz",
        figsize=(14, 5),
        fb_type="gamma")
