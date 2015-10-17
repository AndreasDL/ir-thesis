import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
#stolen with pride from http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Signal_Processing_with_NumPy_Fourier_Transform_FFT_DFT.php

Fs = 150                         # sampling rate

n = len(y)                       # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
freq = frq[range(round(n/2))]           # one side frequency range

Y = np.fft.fft(y)/n              # fft computing and normalization
Y = Y[range(round(n/2))]

print(y)

plt.plot(freq, abs(Y), 'r-')
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

plt.show()

