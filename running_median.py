#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well... running median.

Created on Tue Apr 13 17:00:35 2021
@author: yago
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_running_percentiles(x, y, color='k'):
    N = x.size
    sorted_x = np.argsort(x)
    h = int(np.sqrt(N))
    x_bin = []
    p14 = []
    p50 = []
    p84 = []
    i = 0
    while i+h < N:
        indices = sorted_x[i:i+h]
        x_bin.append(np.nanmedian(x[indices]))
        p14.append(np.nanpercentile(y[indices], 14))
        p50.append(np.nanpercentile(y[indices], 50))
        p84.append(np.nanpercentile(y[indices], 84))
        i += h
    indices = sorted_x[i:]
    # print(i, h, indices, x[indices])
    x_bin.append(np.nanmedian(x[indices]))
    p14.append(np.nanpercentile(y[indices], 14))
    p50.append(np.nanpercentile(y[indices], 50))
    p84.append(np.nanpercentile(y[indices], 84))

    plt.plot(x_bin, p50, ls='-', color=color)
    # plt.plot(x_bin, p14, 'k-', alpha=.25)
    # plt.plot(x_bin, p84, 'k-', alpha=.25)
    plt.fill_between(x_bin, p14, p84, color=color, alpha=.15)


N = 1000
noise = 0.1
x = np.random.rand(N)
y = np.random.normal(loc=x**2, scale=noise, size=N)

plt.plot(x, y, 'k,')
plot_running_percentiles(x, y, 'b')

# %% Bye

print("... Paranoy@ Rulz!")
