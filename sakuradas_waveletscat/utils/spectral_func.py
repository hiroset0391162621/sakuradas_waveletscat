
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sf
import scipy.signal as ssig
from matplotlib.ticker import *
import pandas as pd
from scipy.optimize import curve_fit
import math
from matplotlib import mlab
from matplotlib.colors import Normalize
import matplotlib.dates as dates
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
import matplotlib.ticker as ticker


def smoothTriangle(data,degree,dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""
    triangle=np.array(list(range(degree))+[degree]+list(range(degree))[::-1])+1
    smoothed=[]
    for i in range(degree,len(data)-degree*2):
        point=data[i:i+len(triangle)]*triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals: return smoothed
    smoothed=[smoothed[0]]*(degree+degree//2)+smoothed

    while len(smoothed)<len(data):smoothed.append(smoothed[-1])
    return smoothed


def nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))


def smooth(x, window='boxcar'):
    """ some window smoothing """
    half_win = 11
    window_len = 2*half_win+1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = ssig.boxcar(window_len).astype('complex')
    else:
        w = ssig.hanning(window_len).astype('complex')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[half_win:len(y)-half_win]

def smooth_win(x, half_win=21, window='boxcar'):
    """ some window smoothing
    change length of smoothing window
    """

    window_len = 2*half_win+1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = ssig.boxcar(window_len).astype('complex')
    else:
        w = ssig.hanning(window_len).astype('complex')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[half_win:len(y)-half_win]

def getCoherence(dcs, ds1, ds2):
    # ! number of data
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    # ! calculate coherence
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    coh[valids] = dcs[valids]/(ds1[valids]*ds2[valids])
    coh[coh > (1.0+0j)] = 1.0+0j
    return coh

def cosTaper(windL, percent):
    N = windL
    tp = np.ones(N)
    for i in range(int(N*percent+1)):
        tp[i] *= 0.5 * (1 - np.cos((np.pi * i) / ( N * percent)))

    for i in range(int(N*(1-percent)), N):
        tp[i] *= 0.5 * (1 - np.cos((np.pi * (i+1)) / ( N * percent)))

    return tp


def powerspec(y, sampRate=100, percent_costaper=0.05):
    padd = 2**(nextpow2(len(y))+1)
    freqVec = sf.fftfreq(int(padd), 1./sampRate)[:int(padd)//2]
    tp = cosTaper(len(y), percent_costaper)
    cci = y.copy()
    cci -= np.mean(cci)
    cci *= tp

    Fcur = sf.fft(cci, n=int(padd))[:int(padd)//2]
    Fcur2 = np.real(Fcur)**2 + np.imag(Fcur)**2
    #Fcur2 /= (len(Fcur2)*(1./sampRate))
    #dcur = np.sqrt(smooth(Fcur2, window='hanning'))
    dcurpsd = smooth(Fcur2, window='hanning') / freqVec
    dcurpsd = smoothTriangle(dcurpsd[:int(padd)//2], 10)

    return freqVec, dcurpsd

def spec(y, sampRate=100, percent_costaper=0.05):
    """
    y: numpy array
    """
    padd = 2**(nextpow2(len(y))+1)
    freqVec = sf.fftfreq(int(padd), 1./sampRate)[:int(padd)//2]
    tp = cosTaper(len(y), percent_costaper)
    cci = y.copy()
    cci -= np.mean(cci)
    cci *= tp

    Fcur = sf.fft(cci, n=int(padd))[:int(padd)//2]
    Fcur2 = np.sqrt( np.real(Fcur)**2 + np.imag(Fcur)**2 )
    #Fcur2 /= (len(Fcur2)*(1./sampRate))

    return freqVec, Fcur2


def spec_smooth(y, sampRate=100.0, half_win=21, percent_costaper=0.05):
    """
    y: numpy array
    """
    padd = 2**(nextpow2(len(y))+1)
    freqVec = sf.fftfreq(int(padd), 1./sampRate)[:int(padd)//2]
    tp = cosTaper(len(y), percent_costaper)
    cci = y.copy()
    cci -= np.mean(cci)
    cci *= tp

    Fcur = sf.fft(cci, n=int(padd))[:int(padd)//2]
    Fcur_amp = np.real(Fcur)**2 + np.imag(Fcur)**2
    Fcur_amp /= (len(Fcur_amp)*(1./sampRate))

    Fcur_smooth = smooth_win(Fcur_amp, half_win, window='hanning')

    return freqVec, Fcur_smooth


def whiten_Hirose(y, fmin, fmax, sampRate=100.0, half_win=21, percent_costaper=0.05):
    """
    y: numpy array
    """
    padd = 2**(nextpow2(len(y))+1)
    freqVec = sf.fftfreq(int(padd), 1./sampRate)[:int(padd)/2]
    tp = cosTaper(len(y), percent_costaper)
    cci = y.copy()
    cci -= np.mean(cci)
    cci *= tp

    ### cal spectral amplitude
    Fcur = sf.fft(cci, n=int(padd))[:int(padd)/2]
    Fcur_amp = np.real(Fcur)**2 + np.imag(Fcur)**2
    Fcur_amp /= (len(Fcur_amp)*(1./sampRate))

    ### smooth spectral amplitude
    Fcur_smooth = smooth_win(Fcur_amp, half_win, window='hanning')

    ### whitening
    indRange = np.argwhere(np.logical_and(freqVec >= fmin, freqVec <= fmax))

    Fcur_whiten = Fcur_amp.copy()

    Fcur_whiten[indRange] = Fcur_amp[indRange] / Fcur_smooth[indRange]

    return freqVec, Fcur_whiten

def whitening_outtimeseries(matsign, Nfft, tau, frebas, frehaut, plot=False):
    """This function takes 1-dimensional *matsign* timeseries array,
    goes to frequency domain using fft, whitens the amplitude of the spectrum
    in frequency domain between *frebas* and *frehaut*
    and returns the whitened fft.

    Parameters
    ----------
    matsign : numpy.ndarray
        Contains the 1D time series to whiten
    Nfft : int
        The number of points to compute the FFT
    tau : int
        The sampling frequency of the `matsign`
    frebas : int
        The lower frequency bound
    frehaut : int
        The upper frequency bound
    plot : bool
        Whether to show a raw plot of the action (default: False)



    Returns
    -------
    data : numpy.ndarray
        The FFT of the input trace, whitened between the frequency bounds
    """
    # if len(matsign)/2 %2 != 0:
        # matsign = np.append(matsign,[0,0])

    if plot:
        plt.subplot(411)
        plt.plot(np.arange(len(matsign)) * tau, matsign)
        plt.xlim(0, len(matsign) * tau)
        plt.title('Input trace')

    Napod = 300

    #freqVec = np.arange(0., Nfft / 2.0) / (tau * (Nfft - 1))
    freqVec = sf.fftfreq(Nfft, 1.0/tau)
    J = np.where((freqVec >= frebas) & (freqVec <= frehaut))[0]
    low = J[0] - Napod
    if low < 0:
        low = 0

    porte1 = J[0]
    porte2 = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = Nfft / 2

    FFTRawSign = sf.fft(matsign, Nfft)

    if plot:
        plt.subplot(412)
        axis = np.arange(len(FFTRawSign))
        plt.plot(axis[1:], np.abs(FFTRawSign[1:]))
        plt.xlim(0, max(axis))
        plt.title('FFTRawSign')

    # Apodisation a gauche en cos2
    FFTRawSign[0:low] *= 0
    FFTRawSign[low:porte1] = np.cos(np.linspace(np.pi / 2., np.pi, porte1 - low)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[low:porte1]))
    # Porte
    FFTRawSign[porte1:porte2] = np.exp(
        1j * np.angle(FFTRawSign[porte1:porte2]))
    # Apodisation a droite en cos2
    FFTRawSign[porte2:high] = np.cos(np.linspace(0., np.pi / 2., high - porte2)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[porte2:high]))

    if low == 0:
        low = 1

    FFTRawSign[-low:] *= 0
    # Apodisation a gauche en cos2
    FFTRawSign[-porte1:-low] = np.cos(np.linspace(0., np.pi / 2., porte1 - low)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[-porte1:-low]))
    # Porte
    FFTRawSign[-porte2:-porte1] = np.exp(
        1j * np.angle(FFTRawSign[-porte2:-porte1]))
    # ~ # Apodisation a droite en cos2
    FFTRawSign[-high:-porte2] = np.cos(np.linspace(np.pi / 2., np.pi, high - porte2)) ** 2 * np.exp(
        1j * np.angle(FFTRawSign[-high:-porte2]))

    FFTRawSign[high:-high] *= 0

    FFTRawSign[-1] *= 0.
    if plot:
        plt.subplot(413)
        axis = np.arange(len(FFTRawSign))
        plt.axvline(low, c='g')
        plt.axvline(porte1, c='g')
        plt.axvline(porte2, c='r')
        plt.axvline(high, c='r')

        plt.axvline(Nfft - high, c='r')
        plt.axvline(Nfft - porte2, c='r')
        plt.axvline(Nfft - porte1, c='g')
        plt.axvline(Nfft - low, c='g')

        plt.plot(axis, np.abs(FFTRawSign))
        plt.xlim(0, max(axis))

    wmatsign = np.real(sf.ifft(FFTRawSign))
    del matsign
    if plot:
        plt.subplot(414)
        plt.plot(np.arange(len(wmatsign)) * tau, wmatsign)
        plt.xlim(0, len(wmatsign) * tau)
        plt.show()

    return wmatsign
    #return FFTRawSign





#################
### spec-gram
def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def specgram(data, samp_rate, per_lap=0.9, wlen=None, outfile=None, dbscale=False, mult=8.0, cmap=cm.rainbow, title=None, show=True, clip=[0.0,1.0], zorder=None):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to
        (samp_rate/100.0).
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type sphinx: bool
    :param sphinx: Internal flag used for API doc generation, default False
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    """
    #import matplotlib.pyplot as plt
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = samp_rate / 100.

    npts = len(data)
    # nfft needs to be an integer, otherwise a deprecation will be raised
    # add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                        pad_to=mult, noverlap=nlap)
    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]
    
    vmin, vmax = clip
    # if vmin < 0 or vmax > 1 or vmin >= vmax:
    #     msg = "Invalid parameters for clip option."
    #     raise ValueError(msg)
    # _range = float(specgram.max() - specgram.min())
    # vmin = specgram.min() + vmin * _range
    # vmax = specgram.min() + vmax * _range
    # norm = Normalize(vmin, vmax, clip=True)


    # fig = plt.figure(figsize=(10,6))
    # ax1 = fig.add_axes([0.1,0.1,0.75,0.5])
    # ax2 = fig.add_axes([0.1,0.7,0.75,0.25])
    # cax = fig.add_axes([0.9, 0.1, 0.02, 0.5])

    # t = np.linspace(0,len(data)/samp_rate,len(data))
    # ax2.plot(t, data, color="C7", lw=1)
    # ax2.set_xlim(0, end)
    # ax2.set_ylim(-1e-4, 1e-4)

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # # argument None is not allowed for kwargs on matplotlib python 3.3
    # kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
    #           if v is not None}


    # this method is much much faster!
    specgram = np.flipud(specgram)
    # center bin
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
            freq[0] - halfbin_freq, freq[-1] + halfbin_freq)

    return specgram, extent
    #im = ax1.imshow(specgram, interpolation="nearest", extent=extent, **kwargs)

    # im = ax1.imshow(specgram, interpolation="nearest", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    # sfmt=ticker.ScalarFormatter(useMathText=True)
    # sfmt.set_powerlimits((0, 0))
    # cbar = fig.colorbar(im, cax=cax, orientation='vertical', format=sfmt)
    # cbar.set_clim(vmin ,vmax)



    # # set correct way of axis, whitespace before and after with window
    # # length
    # ax1.axis('tight')
    # ax1.set_xlim(0, end)
    # ax1.set_ylim(0, 20)
    

    # ax1.set_xlabel('Time [s]', fontsize=16)
    # ax1.set_ylabel('Frequency [Hz]', fontsize=16)
    # ax2.set_ylabel('Velocity [m/s]', fontsize=16)

    # # ax1.xaxis.set_major_locator(MultipleLocator(100))
    # # ax1.xaxis.set_minor_locator(MultipleLocator(50))
    # # ax1.yaxis.set_major_locator(MultipleLocator(1))
    # # ax1.yaxis.set_minor_locator(MultipleLocator(0.5))

    # # ax2.xaxis.set_major_locator(MultipleLocator(100))
    # # ax2.xaxis.set_minor_locator(MultipleLocator(50))
    # ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax2.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))



    # if title:
    #     ax2.set_title(title)

    # if outfile:
    #     fig.savefig(outfile, dpi=200)
    #     plt.close()
    # if show:
    #     plt.show()
    # else:
    #     return fig


# if __name__ == '__main__':
#
#     title = str(Year[pp])+str(Month[pp])+str(Day[pp])
#     spectrogram(np.array(rmeanZ_1d), samp_rate=100.0, per_lap=0.9, wlen=None, outfile="test.png", dbscale=False, mult=8.0, cmap=cm.rainbow, title=title, show=True, clip=[0.0,1.0], zorder=None)
