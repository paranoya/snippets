#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the mode of a data sample.

Created on March 2 2022
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


def adaptive_histogram(data, bandwidth=None):
    """
    Compute histogram based on a top-hat kernel with the specified (adaptive) bandwidth.
    :param data: Collection of data points
    :param bandwidth: width of the top-hat kernel
    """
    if bandwidth is None:
        h1 = 1 + int(np.sqrt(data.size))
        h2 = 1 + int(data.size / 2)
        h = 1 + int(np.sqrt(h1 * h2))
    else:
        h = bandwidth
    x = np.sort(data.flatten())
    x_cumul = np.cumsum(x)

    x_mid = (x[h:] + x[:-h]) / 2
    # x_mean = (x_cumul[h:] - x_cumul[:-h]) / h
    half_h = int(h/2)
    x_median = x[half_h:half_h-h]
    density = h / (x[h:] - x[:-h])
    x_bin = np.sqrt(x_mid*x_median)
    return x_bin, density / data.size


# %% Estimate mean and variance
# around peak, assuming Gaussian

# estimated_mean = np.mean(x)
# estimated_variance = np.var(x)
# print(estimated_mean, estimated_variance)
# while True:
#     weight = np.exp(-.5 * (x - estimated_mean) ** 2 / estimated_variance)
#     total_weight = np.sum(weight)
#     mu = np.sum(weight * x) / total_weight
#     sigma2 = np.sum(weight * x * x) / total_weight - mu * mu
#     estimated_mean = (mu * estimated_variance - estimated_mean * sigma2) / (estimated_variance - sigma2)
#     estimated_variance = estimated_variance * sigma2 / (estimated_variance - sigma2)
#     print(mu, 2 * sigma2, total_weight, estimated_mean, estimated_variance, mu - estimated_mean,
#           np.sqrt(estimated_variance / total_weight))
#     if (mu - estimated_mean) ** 2 < .01 * estimated_variance / total_weight:
#         break

def find_mode(x):
    """
    Locate the mode of a distribution by fitting a polynomial within +-one-sigma interval.
    :param x: Collection of data points
    """
    # Legendre polynomials
    def L0(x): return np.ones_like(x)
    def L1(x): return x
    def L2(x): return (3 * x ** 2 - 1) / 2
    # def L3(x): return (5*x**3 - 3*x) / 2
    # def L4(x): return (35*x**4 - 30*x**2 + 3) / 8
    def norm_L(n): return 2 / (2 * n + 1)

    x0 = np.nanmean(x)
    sigma = np.nanstd(x)
    delta_peak = 0
    i = 0
    while True:
        i += 1
        x0 += delta_peak
        delta = x - x0
        delta = delta[np.abs(delta) < sigma] / sigma
        # scalar product:
        c0 = np.mean(L0(delta)) / norm_L(0) # point density · L0
        c1 = np.mean(L1(delta)) / norm_L(1)  # point density · P1
        c2 = np.mean(L2(delta)) / norm_L(2)  # point density · P2
        # c3 = np.mean(L3(delta)) / norm_L(3)  # point density · P3
        # c4 = np.mean(L4(delta)) / norm_L(4)  # point density · P4
        polynomial_fit = c0*L0(delta) + c1*L1(delta) + c2*L2(delta)  # + c3*L3(delta) + c4*L4(delta)
        delta_peak = delta[np.argmax(polynomial_fit)] * sigma/np.sqrt(i)  # divide by sqrt(i) to prevent oscillations
        # print(x0, delta_peak)
        if np.abs(delta_peak) <= sigma/x.size:
            return x0, x0+delta*sigma, polynomial_fit*delta.size/x.size/sigma


# %% Generate data

N = 10000

D = 2
distribution = 'Gaussian in {} dimensions'.format(D)
x = 0
for d in range(D):
    x += np.random.normal(0., 1., N)**2
x = np.sqrt(x)
sorted_x = np.sort(x)
true_density = sorted_x**(D-1) * np.exp(-.5 * sorted_x**2) *2**(1-D/2)/gamma(D/2)
true_mode = np.sqrt(D-1)

# %% Find mode

x0, x_fit, polynomial_fit = find_mode(x)
print("Mode = {:.3f} (true = {:.3f})".format(x0, true_mode))

# %% Identify outliers

N_mode = np.searchsorted(sorted_x, x0)
N_normal = 2*N_mode
if N_normal < N:
    print('High values are unusual:')
    threshold = sorted_x[N_normal]
    contamination = np.searchsorted(sorted_x, x0-(threshold-x0))
else:
    print('Low values are unusual:')
    N_normal = 2*(N-N_mode)
    threshold = sorted_x[-N_normal]
    contamination = N - np.searchsorted(sorted_x, x0 + (x0-threshold))
print(f'{N_normal}/{N} "normal" values (threshold = {threshold:.3f}; contamination = {contamination}/{N - N_normal})')

# %% Plot figure

# plt.ion()
fig, ax = plt.subplots()
ax.set_title('{} - {} points'.format(distribution, N))
ax.set_xlabel('x')
ax.set_ylabel('probability')

ax.plot(sorted_x, true_density, 'k--')
ax.axvline(true_mode, c='k', ls='--', label='True solution: {:.3f}'.format(true_mode))

xx, den = adaptive_histogram(x)
adaptive_peak = xx[np.argmax(den)]
ax.plot(xx, den, 'b-')
ax.axvline(adaptive_peak, c='b', ls=':', label='Adaptive histogram: {:.3f}'.format(adaptive_peak))

ax.plot(x_fit, polynomial_fit, 'r.')
ax.axvline(x0, c='r', ls='-.', label='Legendre fit: {:.3f}'.format(x0))
ax.axvline(threshold, c='r', ls='-', label='threshold: {:.3f}'.format(threshold))

ax.hist(x, bins=np.linspace(np.min(x), np.max(x), 2+int(np.sqrt(x.size))), density=True, alpha=.2)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(.5/N, 1.1)
# ax.set_yscale('log')
ax.legend()

plt.show()

# %% Bye

print("... Paranoy@ Rulz!")
