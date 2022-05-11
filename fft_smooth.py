#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute a smooth histogram of a data sample.

Created on March 11, 2022
@author: yago
"""

from __future__ import print_function, division

from matplotlib import use
import numpy as np
from math import gamma

try:
    use('qt5Agg')
except:
    pass
from matplotlib import pyplot as plt

from mode import adaptive_histogram, find_mode


if __name__ == "__main__":
    # %% Generate data

    N = 1000

    D = 2
    distribution = 'Gaussian in {} dimensions'.format(D)
    x = 0
    for d in range(D):
        x += np.random.normal(0., 1., N)**2
    x = np.sqrt(x)
    sorted_x = np.sort(x)
    delta_x = sorted_x[1:]-sorted_x[:-1]
    delta_x = np.log10(delta_x)
    true_density = sorted_x**(D - 1) * np.exp(-.5 * sorted_x**2) * 2 ** (1 - D / 2) / gamma(D / 2)

    # %% Analysis

    fft_delta_x = np.fft.rfft(delta_x)
    amplitude = np.abs(fft_delta_x)
    phase = np.angle(fft_delta_x)
    sorted_amplitude = np.sort(amplitude)

    amplitude_bin, density = adaptive_histogram(amplitude)
    x0, x_fit, polynomial_fit = find_mode(amplitude)
    N_mode = np.searchsorted(sorted_amplitude, x0)
    N_normal = 2*N_mode
    if N_normal < N:
        print('High values are unusual:')
        # threshold = sorted_amplitude[N_normal]
        # contamination = np.searchsorted(sorted_amplitude, x0-(threshold-x0))
    else:
        print('Low values are unusual:')
        N_normal = 2*(N-N_mode)
        threshold = sorted_amplitude[-N_normal]
        contamination = N - np.searchsorted(sorted_amplitude, x0 + (x0-threshold))
    print(f'{N_normal}/{N} "normal" values (threshold = {threshold:.3f}; contamination = {contamination}/{N - N_normal})')

    # %% Plot figure

    # plt.ion()
    fig, ax = plt.subplots()
    ax.set_title('{} - {} points'.format(distribution, N))

    # ax.plot(amplitude_bin, density, 'c-')
    # ax.plot(x_fit, polynomial_fit, 'r.')
    # ax.axvline(x0, c='r', ls='-.', label='Legendre fit: {:.3f}'.format(x0))
    # ax.axvline(threshold, c='r', ls='-', label='threshold: {:.3f}'.format(threshold))
    # ax.hist(amplitude,
    #         bins=np.linspace(np.min(amplitude), np.max(amplitude), 2 + int(np.sqrt(amplitude.size))),
    #         density=True, alpha=.2)

    ax.plot(amplitude, 'c-')
    ax.axhline(x0, ls=':')
    ax.axhline(threshold, ls=':')

    # ax.set_xlabel('x')
    # ax.set_ylabel('probability')
    #
    # ax.plot(sorted_x, true_density, 'k--')
    #

    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(.5 / N, 1.1)
    ax.set_xscale('log')
    # # ax.set_yscale('log')
    # ax.legend()

    plt.show()

    # %% Bye

    print("... Paranoy@ Rulz!")
