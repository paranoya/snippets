#!/usr/bin/python

from __future__ import print_function, division

from matplotlib import use
use('qt5Agg')

from matplotlib import pyplot as plt
import time
import numpy as np

plt.ion()
plt.show()

tstart = time.time()  # for profiling
x = np.arange(0, 2*np.pi, 0.01)  # x-array
line, = plt.plot(x, np.sin(x))
for i in np.arange(1, 200):
    line.set_ydata(np.sin(x+i/10.0))  # update the data
    plt.draw()  # redraw the canvas
    plt.pause(0.001)

print('FPS:', 200/(time.time()-tstart))
