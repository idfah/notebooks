"""Bandpass filters.
by
  Elliott Forney
  2013
"""

import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt


class IIRFilter(object):
    """IIR bandpass filter.
    """
    def __init__(self, lowFreq, highFreq, sampRate=1, order=3,
                 ftype='butter', zeroPhase=True, **kwargs):
        """Construct a new IIR filter.
        """
        self.lowFreq   = lowFreq
        self.highFreq  = highFreq
        self.sampRate  = sampRate
        self.order     = order
        self.ftype     = ftype
        self.zeroPhase = zeroPhase

        self.nyquist = sampRate * 0.5

        self.low = lowFreq / self.nyquist
        if self.low > 1.0:
            raise Exception("lowFreq=%d is above nyquist rate." % lowFreq)
        if self.low < 0.0:
            raise Exception("lowFreq=%d is less than zero." % lowFreq)

        self.high = highFreq / self.nyquist
        if self.high != np.Inf:
            if self.high > 1.0:
                raise Exception("highFreq=%d is above nyquist rate." % highFreq)
            if self.high < 0.0:
                raise Exception("highFreq=%d is less than zero." % highFreq)

        if self.low == 0.0 and self.high != np.Inf:
            self.btype = 'lowpass'
            self.Wn = self.high
        elif self.low > 0.0 and self.high == np.Inf:
            self.btype = 'highpass'
            self.Wn = self.low
        elif self.low > 0.0 and self.high != np.Inf:
            self.btype = 'bandpass'
            self.Wn = (self.low,self.high)
        else:
            raise Exception("Invalid filter corners.")

        self.b, self.a = spsig.iirfilter(order, self.Wn,
            ftype=ftype, btype=self.btype, **kwargs)

        self.zi = spsig.lfilter_zi(self.b, self.a)

    def filter(self, data, axis=0):
        """Filter new data.
        """
        if self.zeroPhase:
            return spsig.filtfilt(self.b, self.a, data, axis=axis)

        else:
            ziShape = [1,] * data.ndim
            ziShape[axis] = self.zi.size
            zi = self.zi.reshape(ziShape)

            s = [slice(None),] * data.ndim
            s[axis] = 0
            d0 = data[s]
            d0Shape = list(data.shape)
            d0Shape[axis] = 1
            d0 = d0.reshape(d0Shape)

            return spsig.lfilter(self.b, self.a, data, axis=axis, zi=zi*d0)

    def plotFreqResponse(self, drawCorners=True, ax=None, **kwargs):
        """Plot the frequency response of the filter.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Gain (Amplitude$^2$)')

        w, h = spsig.freqz(self.b, self.a, worN=1024)

        lines = ax.plot((self.sampRate * 0.5 / np.pi) * w, np.abs(h),
                        label='Frequency Response', **kwargs)

        if drawCorners:
            ax.hlines(np.sqrt(0.5), 0, 0.5*self.sampRate,
                color='red', linestyle='-.', label='Half Power')
            ax.vlines((self.lowFreq,self.highFreq), 0, 1,
                color='violet', linestyle='--', label='Corners')

        #ax.legend(loc='best')

        return {'ax': ax, 'lines': lines}

def IIRFilterDemo():
    bp = IIRFilter(50.0,70.0, sampRate=256, ftype='butter', order=1)
    ax = bp.plotFreqResponse(drawCorners=False)['ax']

    bp = IIRFilter(50.0,70.0, sampRate=256, ftype='butter', order=2)
    bp.plotFreqResponse(drawCorners=False, ax=ax)

    bp = IIRFilter(50.0,70.0, sampRate=256, ftype='butter', order=5)
    bp.plotFreqResponse(drawCorners=True, ax=ax)

    ax.set_title('Filter Frequency Response')
    ax.legend(['order 1', 'order 2', 'order 5'], loc='best')

    freq  = np.array([8.0, 14.0, 20.0, 30.0, 40.0, 60.0])
    phase = np.arange(freq.shape[0])
    amp = np.linspace(1,0,freq.shape[0])

    time = np.linspace(0.0, 3.0, 3.0*256)

    wave = amp[None,:] * np.cos(2.0 * np.pi * freq[None,:] * time[:,None] + phase[None,:])
    wave = np.sum(wave, axis=1)

    bp = IIRFilter(2.0, 25.0, sampRate=256, order=3)
    filtWave = bp.filter(wave)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(time, wave, linewidth=2, color='gray')
    ax.plot(time, filtWave, linewidth=2, color='red')
    ax.legend(['raw', 'filtered'])

    plt.show()

if __name__ == '__main__':
    IIRFilterDemo()
