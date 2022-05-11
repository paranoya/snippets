#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the mode of a data sample.

Created on March 2, 2022
@author: Yago Ascasibar
"""

from __future__ import print_function, division

from matplotlib import use
try:
    use('qt5Agg')
except:
    pass
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits


def find_threshold(x, weights=None, max_steps=100):
    """
    Locate the threshold to separate 'sources' from a 'background' population in a data collection,
    assuming that the background probability distribution is symmetric around a pivot point.

    :param x: Collection of data points.
    :param weights: Statistical weight of each point.
    :param max_steps: Maximum number of steps when looking for the optimal pivot.

    :output optimal_threshold: Returns optimal threshold to separate sources.
    """
    if weights is None:
        weights = np.ones_like(x)
    sorted_by_x = np.argsort(x.flatten())
    sorted_x = x.flatten()[sorted_by_x]
    sorted_weight = weights.flatten()[sorted_by_x]
    cumulative_mass = np.nancumsum(sorted_weight)
    total_mass = cumulative_mass[-1]
    n_data = sorted_x.size

    symmetric_mass = cumulative_mass.copy()
    x_pivot = []
    m_pivot = []
    excess = []
    x_threshold = []
    contamination = []
    median_mass = np.searchsorted(cumulative_mass/total_mass, .5)
    for pivot in np.arange(median_mass-1, 1, -int(n_data/max_steps)-1):
        x0 = sorted_x[pivot]
        pivot_mass = cumulative_mass[pivot]
        # print(pivot, x0, pivot_mass)
        x_pivot.append(x0)
        m_pivot.append(pivot_mass)
        symmetric_x = x0 - (sorted_x[pivot+1:] - x0)
        symmetric_mass[pivot+1:] = 2*pivot_mass - np.interp(symmetric_x, sorted_x, cumulative_mass)
        excess.append(np.max(symmetric_mass-cumulative_mass))
        x_threshold.append(np.interp(2*pivot_mass, cumulative_mass, sorted_x))
        contamination.append(np.interp(2*x0-x_threshold[-1], sorted_x, cumulative_mass))
    x_pivot = np.array(x_pivot)
    m_pivot = np.array(m_pivot)
    excess = np.array(excess)
    x_threshold = np.array(x_threshold)
    contamination = np.array(contamination)
    source_mass = total_mass-2*m_pivot
    metric = (contamination+excess)/source_mass

    contamination /= source_mass
    contamination_min = np.nanargmin(contamination)
    contamination_max = contamination_min + np.nanargmax(contamination[contamination_min:])
    # print(contamination_min, contamination_max)

    '''
    plt.plot(x_pivot, source_mass/total_mass, 'k-')
    # plt.plot(x_pivot, excess/source_mass, 'r:')
    plt.plot(x_pivot, contamination/source_mass, 'r--')
    plt.axvline(x_pivot[contamination_min], c='b', ls=':')
    plt.plot(x_pivot, metric, 'r-')
    plt.axvline(x_pivot[contamination_max], c='b', ls=':')
    # plt.axvline(x_optimal, c='r', ls='-')
    plt.yscale('log')
    plt.show()
    '''

    # %% Refined estimate

    symmetric_mass = cumulative_mass.copy()
    x_pivot = np.linspace(x_pivot[contamination_min], x_pivot[contamination_max], max_steps)
    m_pivot = []
    excess = []
    x_threshold = []
    contamination = []
    for x0 in x_pivot:
        pivot = np.searchsorted(sorted_x, x0)
        pivot_mass = np.interp(x0, sorted_x, cumulative_mass)
        # print(x0, pivot_mass)
        m_pivot.append(pivot_mass)
        symmetric_x = x0 - (sorted_x[pivot+1:] - x0)
        symmetric_mass[pivot+1:] = 2*pivot_mass - np.interp(symmetric_x, sorted_x, cumulative_mass)
        excess.append(np.max(symmetric_mass-cumulative_mass))
        x_threshold.append(np.interp(2*pivot_mass, cumulative_mass, sorted_x))
        contamination.append(np.interp(2*x0-x_threshold[-1], sorted_x, cumulative_mass))
    m_pivot = np.array(m_pivot)
    excess = np.array(excess)
    x_threshold = np.array(x_threshold)
    contamination = np.array(contamination)

    source_mass = total_mass-2*m_pivot
    metric = (contamination+excess)/source_mass
    pivot_optimal = np.argmin(metric)
    x_optimal = x_pivot[pivot_optimal]
    threshold_optimal = x_threshold[pivot_optimal]
    print(f'refined optimal: mode={x_optimal} -> threshold={threshold_optimal}')

    plt.plot(x_pivot, source_mass/total_mass, 'k-')
    plt.plot(x_pivot, excess/source_mass, 'r:')
    plt.plot(x_pivot, contamination/source_mass, 'r--')
    plt.plot(x_pivot, metric, 'r-')
    plt.axvline(x_optimal, c='r', ls='-')

    plt.show()

    # plt.plot(sorted_x, cumulative_mass, 'b-')
    # plt.plot(sorted_x, symmetric_mass, 'k-')
    # plt.show()

    return threshold_optimal


if __name__ == "__main__":
    # %% Read data
    hdu = fits.open('data/CIG_335.fits')
    x = hdu[0].data[3000:4000, 1500:2500] * 1.  # to make sure it's converted to float

    # %% Find threshold

    threshold_optimal = find_threshold(x)

    # %% Bye

    print("... Paranoy@ Rulz!")
