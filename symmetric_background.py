#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the mode of a data sample.

Created on March 2, 2022
@author: Yago Ascasibar
"""

from __future__ import print_function, division

import pylab as pl
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
# from skimage.filters import threshold_multiotsu
from photutils import segmentation as segm

if __name__ == "__main__":
    # %% Read data

    hdu = fits.open('data/CIG_335.fits')
    x = hdu[0].data[3000:4000, 1500:2500] * 1.  # to make sure it's converted to float
    # hdu = fits.open('data/hcg44_cube_R.fits')
    # x = hdu[0].data[69]*1.

    # %% Find threshold ------------------------------

    # if weights is None:
    weights = np.ones_like(x)
    sorted_by_x = np.argsort(x.flatten())
    sorted_x = x.flatten()[sorted_by_x]
    sorted_weight = weights.flatten()[sorted_by_x]
    cumulative_mass = np.nancumsum(sorted_weight)
    sorted_weight /= cumulative_mass[-1]
    cumulative_mass /= cumulative_mass[-1]

    n_data = sorted_x.size
    sqrt_n = np.sqrt(n_data)
    delta = .5

    m = np.linspace(.25, .75, int(sqrt_n))
    x_top = np.interp((1+delta)*m, cumulative_mass, sorted_x)
    x_mid = np.interp(m, cumulative_mass, sorted_x)
    x_bot = np.interp((1-delta)*m, cumulative_mass, sorted_x)
    rho_top = delta * m / (x_top - x_mid)
    rho_bot = delta * m / (x_mid - x_bot)
    peak = np.nanargmin((rho_top - rho_bot) ** 2)
    x0 = x_mid[peak]
    M_background = 2 * m[peak]
    M_signal = 1 - M_background

    M_above = 1 - cumulative_mass
    M_symmetric_above = np.interp(2 * x0 - sorted_x, sorted_x, cumulative_mass, left=0.)
    left = np.where(sorted_x < x0)
    M_symmetric_above[left] = M_background - cumulative_mass[left]
    M_signal_above = M_above - M_symmetric_above
    purity = M_signal_above / (M_above+1e-30)
    purity[-1] = 1
    mean_purity_below = np.cumsum(sorted_weight*purity)/cumulative_mass
    mean_purity_above = (np.cumsum((sorted_weight*purity)[::-1])/np.cumsum(sorted_weight[::-1]))[::-1]

    a = np.interp(M_signal+M_background/4, purity, sorted_x)
    b = np.interp(1-M_background/4, purity, sorted_x)
    x_background = a - (b-a)/2
    x_signal = b + (b-a)/2
    purity_background = np.interp(x_background, sorted_x, purity)
    purity_signal = np.interp(x_signal, sorted_x, purity)

    # %% Segmentation ------------------------------

    purity_map = np.interp(x, sorted_x, purity)
    # pmap = np.empty_like(x)
    # for pix, pur in zip(sorted_by_x, purity):
    #     pmap[np.unravel_index(pix, x.shape)] = pur

    dirty_segmentation = segm.detect_sources(x, x_signal, npixels=1)
    dirty_catalog = segm.SourceCatalog(x, dirty_segmentation)
    dirty_area = np.sum(dirty_catalog.area)
    sorted_dirty_area = np.sort(dirty_catalog.area)
    cumulative_dirty_area = np.cumsum(sorted_dirty_area)
    minimum_size = np.ceil(np.interp((1-purity_signal)*dirty_area, cumulative_dirty_area, sorted_dirty_area).to_value())
    print(f'{len(dirty_catalog)} sources ({dirty_area}) found above threshold {x_signal:.3g};',
          f'purity: {purity_signal*100:.2f}%')
    print(f'Expected contamination: {(1-purity_signal)*dirty_area:.1f} => minimum size = {minimum_size}')
    noise_segmentation = segm.detect_sources(x0-x, x_signal-x0, npixels=1)
    noise_catalog = segm.SourceCatalog(x0-x, noise_segmentation)
    noise_area = np.sum(noise_catalog.area)
    minimum_size = max(minimum_size, np.ceil(np.max(noise_catalog.area.to_value())))
    print(f'{len(noise_catalog)} spurious sources ({noise_area}) found below threshold {2*x0-x_signal:.3g}')
    print(f'=> minimum size updated to {minimum_size}')
    clean_segmentation = segm.detect_sources(x, x_signal, npixels=minimum_size)
    clean_catalog = segm.SourceCatalog(x, clean_segmentation)

    sorted_noise_area = np.sort(noise_catalog.area)
    cumulative_noise_area = np.cumsum(sorted_noise_area)

    # %% Plot result ------------------------------
    plt.ion()

    '''
    plt.figure()
    plt.plot(m, x_top, 'k:', label=f'{(1+delta)*100:g} %')
    plt.plot(m, x_mid, 'k-', label=f'100 %')
    plt.plot(m, x_bot, 'k:', label=f'{(1-delta)*100:g} %')

    plt.xlabel('Mass fraction')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.show()
    '''

    # '''
    plt.figure()
    plt.plot(sorted_x, M_above, 'k-', label='total')
    plt.plot(sorted_x, M_symmetric_above, 'b-', label='background')
    plt.plot(sorted_x, M_signal_above, 'r-', label='signal')
    plt.axvline(x0, c='k', ls=':')
    plt.axvline(x_signal, c='k', ls='--')

    plt.xlabel('x')
    pl.xlim(3*x0-3*x_signal, 5*x_signal-x0)
    plt.ylabel('Mass above')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()
    # '''

    '''
    plt.figure()
    plt.plot(purity, cumulative_mass)

    plt.xlabel('purity')
    plt.xscale('log')
    plt.ylabel('probability density')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()
    '''

    # '''
    plt.figure()
    plt.hist(sorted_x, bins=np.linspace(3*x0-3*x_signal, 5*x_signal-x0, 501), density=True)
    plt.axvline(2*x0-x_signal, c='k', ls='--')
    plt.axvline(x0, c='k', ls=':')
    plt.axvline(x_background, c='k', ls='--')
    plt.axvline(x_signal, c='k', ls='--')

    plt.xlabel('intensity')
    plt.ylabel('probability density')
    plt.yscale('log')
    plt.grid()
    plt.show()
    # '''

    # '''
    plt.figure()
    plt.hist(purity, bins=np.linspace(0, 1, 501), density=True)
    plt.axvline(purity_background, c='k', ls='--')
    plt.axvline(purity_signal, c='k', ls='--')
    # hist = plt.hist(np.log10(purity), bins=np.linspace(-2, 0, 101), density=True)
    # plt.axvline(np.log10(purity_background), c='k', ls='--')
    # plt.axvline(np.log10(purity_signal), c='k', ls='--')

    plt.xlabel('purity')
    plt.ylabel('probability density')
    # plt.ylim(1/n_data, np.max(hist[0]))
    plt.yscale('log')
    # plt.legend()
    plt.grid()
    plt.show()
    # '''

    # '''
    plt.figure()
    plt.plot(sorted_x, purity, 'k-')
    # plt.plot(sorted_x, 1-purity, 'k--')
    # plt.plot(sorted_x, mean_purity_below, 'b:')
    # plt.plot(sorted_x, mean_purity_above, 'r:')
    # plt.plot(sorted_x, (mean_purity_above-mean_purity_below)**2, 'k:')
    plt.plot(a, M_signal+M_background/4, 'k+')
    plt.plot(b, 1-M_background/4, 'k+')
    plt.axvline(x0, c='k', ls=':')
    plt.axvline(x_background, c='k', ls='--')
    plt.axvline(x_signal, c='k', ls='--')
    plt.plot([sorted_x[0], x_background, x_signal, sorted_x[-1]], [M_signal, M_signal, 1, 1], 'k:')

    plt.xlabel('intensity')
    plt.xlim(2*x0-x_signal, 2*x_signal-x_background)
    plt.ylabel('purity')
    # plt.yscale('log')
    # plt.legend()
    # plt.grid()
    plt.show()
    # '''

    # '''
    plt.figure()
    plt.plot(sorted_noise_area, cumulative_noise_area, 'r:', label='noise')
    plt.plot(sorted_dirty_area, cumulative_dirty_area, 'k-', label='signal')
    plt.plot(minimum_size, (1-purity_signal)*dirty_area, 'ko')

    plt.xlabel('Number of pixels')
    plt.ylabel('Region size')
    plt.legend()
    plt.grid()
    plt.show()
    # '''

    # '''
    fig, ax = plt.subplots()
    ax.set_title(f'Clean catalog: {len(clean_catalog)} sources')
    im = ax.imshow(x,
                   interpolation='nearest', origin='lower', cmap='terrain',
                   vmin=2 * x0 - x_signal, vmax=2 * x_signal - x_background
                   )
    ax.plot(clean_catalog.xcentroid, clean_catalog.ycentroid, 'k+', lw=2)
    clean_catalog.plot_kron_apertures((1.0, 1.0), axes=ax, color='black', lw=2)
    plt.plot(clean_catalog.xcentroid, clean_catalog.ycentroid, 'k+')
    cb = fig.colorbar(im)
    cb.ax.axhline(x_background, c='k', ls='--')
    cb.ax.axhline(x_signal, c='k', ls='--')
    cb.ax.axhline(x0, c='k', ls=':')
    fig.show()
    # '''

    '''
    plt.figure()
    plt.imshow(
        np.digitize(x, [x_background, x_signal]),
        # np.digitize(purity_map, threshold_multiotsu(purity)),
        interpolation='nearest', origin='lower', cmap='nipy_spectral')
    # cb = plt.colorbar()
    plt.show()
    '''

    # '''
    plt.figure()
    plt.title(f'Dirty catalog: {len(dirty_catalog)} sources')
    plt.imshow(dirty_segmentation,
               interpolation='nearest', origin='lower', cmap=dirty_segmentation.make_cmap(seed=123)
               )
    # cb = plt.colorbar()
    plt.show()
    # '''

    # '''
    fig, ax = plt.subplots()
    ax.set_title(f'Clean catalog: {len(clean_catalog)} sources')
    ax.imshow(clean_segmentation,
              interpolation='nearest', origin='lower', cmap=clean_segmentation.make_cmap(seed=123)
              )
    ax.plot(clean_catalog.xcentroid, clean_catalog.ycentroid, 'w+', lw=2)
    clean_catalog.plot_kron_apertures((1.0, 1.0), axes=ax, color='white', lw=2)
    fig.show()
    # '''

    '''
    plt.figure()
    plt.imshow(purity_map,
               interpolation='nearest', origin='lower', cmap='terrain')
    cb = plt.colorbar()
    cb.ax.axhline(purity_background, c='k', ls='--')
    cb.ax.axhline(purity_signal, c='k', ls='--')
    plt.show()
    '''

    '''
    plt.figure()
    for delta in np.linspace(.25, .75, 3):
        x_top = np.interp((1 + delta) * m, cumulative_mass, sorted_x)
        x_bot = np.interp((1 - delta) * m, cumulative_mass, sorted_x)
        rho_top = delta*m / (x_top - x_mid)
        rho_bot = delta*m / (x_mid - x_bot)
        rho_arit = 2*delta*m / (x_top - x_bot)
        rho_geom = 2*rho_bot*rho_top/(rho_bot+rho_top)
        peak = np.nanargmin((rho_top-rho_bot)**2)
        peak_arit = np.nanargmax(rho_arit)
        peak_geom = np.nanargmax(rho_geom)
        # print(delta, rho_top[peak], rho_bot[peak], rho_arit[peak_arit], rho_geom[peak_geom])
        print(delta, m[peak], m[peak_arit], m[peak_geom])

        plt.plot(m, rho_top, 'r-', alpha=delta, label='top')
        plt.plot(m, rho_bot, 'b-', alpha=delta, label='bottom')
        plt.plot(m, rho_arit, 'k-', alpha=delta, label='weighted')
        plt.plot(m, rho_geom, 'k:', alpha=delta, label='weighted')
        plt.axvline(m[peak], c='g', alpha=delta, ls='--')
        plt.axvline(m[peak_arit], c='k', alpha=delta, ls='-')
        plt.axvline(m[peak_geom], c='k', alpha=delta, ls=':')

    plt.xlabel('Mass fraction')
    plt.ylabel('Density')
    plt.legend()
    # plt.grid()
    plt.show()
    '''

    '''
    plt.figure()
    for delta in np.linspace(.25, .75, 3):
        x_top = np.interp((1 + delta) * m, cumulative_mass, sorted_x)
        x_bot = np.interp((1 - delta) * m, cumulative_mass, sorted_x)
        rho_top = m / (x_top - x_mid)
        rho_bot = m / (x_mid - x_bot)
        peak = np.nanargmin((rho_top-rho_bot)**2)
        print(delta, peak, m[peak], x_mid[peak])
        plt.plot(m, rho_top-rho_bot, label=f'{delta:.3g} {m[peak]:.3g} {x_mid[peak]:.3g}')
    plt.xlabel('Mass fraction')
    plt.ylabel('top-bottom')
    plt.legend()
    plt.grid()
    plt.show()
    '''

    '''
    plt.figure()
    for delta in np.linspace(.25, .75, 3):
        x_top = np.interp((1 + delta) * m, cumulative_mass, sorted_x)
        x_bot = np.interp((1 - delta) * m, cumulative_mass, sorted_x)
        rho_top = m / (x_top - x_mid)
        rho_bot = m / (x_mid - x_bot)
        peak = np.nanargmin((rho_top-rho_bot)**2)

        x0 = x_mid[peak]
        M_background = 2 * m[peak]
        M_signal = 1 - M_background

        M_above = 1 - cumulative_mass
        M_symmetric_above = np.interp(2 * x0 - sorted_x, sorted_x, cumulative_mass, left=0.)
        left = np.where(sorted_x < x0)
        M_symmetric_above[left] = M_background - cumulative_mass[left]
        M_signal_above = M_above - M_symmetric_above
        purity = M_signal_above / M_above
        completeness = M_signal_above / M_signal
        product = purity * completeness
        n_threshold = np.nanargmax(product)
        threshold = sorted_x[n_threshold]

        plt.plot(sorted_x, purity, 'r-', alpha=delta, label=f'purity {delta:.3f}')
        plt.plot(sorted_x, completeness, 'b-', alpha=delta, label=f'completeness {delta:.3f}')
        plt.plot(sorted_x, product, 'k-', alpha=delta, label=f'product {delta:.3f}')
        plt.axvline(threshold, c='k', alpha=delta, ls='--')

    plt.xlabel('Mass fraction')
    plt.ylabel('top-bottom')
    plt.legend()
    plt.grid()
    plt.show()
    '''

    '''
    delta = 1
    m = np.linspace(1/sqrt_n, .5, int(sqrt_n))

    x_top = np.interp((1+delta)*m, cumulative_mass, sorted_x)
    x_mid = np.interp(m, cumulative_mass, sorted_x)
    x_bot = np.interp((1-delta)*m, cumulative_mass, sorted_x)
    plt.plot(m,rho_top-rho_bot, 'k-', label='top-bottom')
    '''

    # %% Bye

    print("... Paranoy@ Rulz!")
