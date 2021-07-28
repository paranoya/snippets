#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a colour map of mass percentiles.

Created on Tue Mar 23 11:34:14 2021
@author: yago
"""

import numpy as np
from matplotlib import pyplot as plt


# %% Original density (e.g. sum of two Gaussians)

x = np.linspace(-3, 3, 101)
y = np.linspace(-3, 3, 101)
X, Y = np.meshgrid(x, y)

mu_x = 1
mu_y = 0
sigma_x = 1.2
sigma_y = .6
density = np.exp(
    - 0.5*((X-mu_x)/sigma_x)**2
    - 0.5*((Y-mu_y)/sigma_y)**2
    ) / (2*np.pi * sigma_x * sigma_y)

mu_x = -1
mu_y = -2
sigma_x = .2
sigma_y = .3
density += np.exp(
    - 0.5*((X-mu_x)/sigma_x)**2
    - 0.5*((Y-mu_y)/sigma_y)**2
    ) / (2*np.pi * sigma_x * sigma_y)


# plt.imshow(density, cmap='terrain')
plt.contourf(X, Y, density, 100, cmap='terrain')
plt.colorbar()
plt.title(r'2D density distribution $\rho(x, y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% Mass within each bin

dx = (x[2:]-x[:-2])/2
dx = np.hstack([dx[0], dx, dx[-1]])
dy = (y[2:]-y[:-2])/2
dy = np.hstack([dy[0], dy, dy[-1]])

mass_histogram = density * dx[np.newaxis, :] * dy[:, np.newaxis]

# plt.imshow(mass_histogram, cmap='terrain')
plt.contourf(X, Y, mass_histogram, 100, cmap='terrain')
plt.colorbar()
plt.title('mass within each bin')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %% Cumulative mass below a certain density

sorted_flat = np.argsort(density.flatten())
density_sorted = density.flatten()[sorted_flat]
cumulative_mass = np.cumsum(mass_histogram.flatten()[sorted_flat])

plt.plot(density_sorted, cumulative_mass)
plt.xlabel('density')
plt.ylabel('cumulative mass')
plt.show()


# %% Get mass frfaction contours

fraction_sorted = cumulative_mass/cumulative_mass[-1]
fraction = np.interp(density, density_sorted, fraction_sorted)

# plt.imshow(fraction, cmap='terrain')
contours = plt.contour(X, Y, fraction, levels=[.1, .5, .9],
                       colors='black', linestyles=['dotted', 'solid', 'dotted'])
# plt.clabel(contours, inline=True, fontsize=8)
plt.contourf(X, Y, fraction, 100, cmap='terrain')
plt.colorbar()
plt.title(r'mass fraction (contours enclose 10, 50, and 90%)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %% Final plot

contours = plt.contour(X, Y, fraction, levels=[.1, .5, .9],
                       colors='black', linestyles=['dotted', 'solid', 'dotted'])
plt.contourf(X, Y, density, 100, cmap='terrain')
plt.colorbar()
plt.title(r'density (contours enclose 10, 50, and 90% mass)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %% Bye

print("... Paranoy@ Rulz!")
