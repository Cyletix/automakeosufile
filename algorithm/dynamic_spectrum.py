'''
Description: 这个不知道怎么换成自己的音频,长度不知道怎么确定,
Author: Cyletix
Date: 2022-06-01 19:07:54
LastEditTime: 2023-03-17 04:40:14
FilePath: \AutoMakeosuFile\dynamic_spectrum.py
'''
#!/usr/bin/env python
# 
# (c) 2017 Juha Vierinen
import matplotlib.pyplot as plt
import numpy as n
import scipy.signal as s

plt.style.use('dark_background')#设置plot风格


# create dynamic spectrum
def spectrogram(x,M=1024,N=128,delta_n=100):
    max_t=int(n.floor((len(x)-N)/delta_n))
    t=n.arange(max_t)
    X=n.zeros([max_t,M],dtype=n.complex64)
    w=s.hann(N)
    xin=n.zeros(N)
    for i in range(max_t):
        xin[0:N]=x[i*delta_n+n.arange(N)]
        X[i,:]=n.fft.fft(w*xin,M)
    return(X)

# sample rate (Hz)
fs=4096.0

# sample indexes (one second of signal)
nn=n.arange(4096)
# generate a chirp signal
x=n.sin(0.15e-14*nn**5.0)

# time step
delta_n=25
M=2048
# create dynamic spectrum.
# Use
# - 2048 point FFT
# - 128 samples for each spectra
# - 100 sample increments in time
S=spectrogram(x,M=M,N=128,delta_n=delta_n)
freqs=n.fft.fftfreq(2048,d=1.0/fs)
time=delta_n*n.arange(S.shape[0])/fs



# plot signal
plt.figure(figsize=(12,10))
plt.subplot(211)
plt.plot(nn/fs,x)
plt.title("Signal $x[n]$")
plt.xlabel("Time (s)")
plt.ylabel("Signal amplitude")

plt.subplot(212)
plt.title("Spectrogram")
plt.pcolormesh(time,freqs[0:(M//2)],n.transpose(10.0*n.log10(n.abs(S[:,0:(M//2)])**2.0)),vmin=0)
plt.xlim([0,n.max(time)])
plt.ylim([0,fs/2.0])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
cb=plt.colorbar(orientation="horizontal")
cb.set_label("dB")
plt.tight_layout()
plt.savefig("dynspec.png")
plt.show()