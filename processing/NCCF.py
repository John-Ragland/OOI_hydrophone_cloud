'''
NCCF.py - building methods that utilizide Dask for calculate cross-correlation estimates

Pre-processing methods will hopefully be modular eventually 

'''

from scipy import signal
import scipy
import xarray as xr
import numpy as np
from numpy import matlib

def preprocess(da, W=30, Fs=200, tide=False, tide_interp=None):
    '''
    preprocess - takes time series and performs pre-processing steps for estimating cross-correlation

    Currently pre-processing is fixed to be
    - divide into 30 s segments
    - filter between 1 and 90 Hz
    - clip to 3 times std for every chunk (which is also 1 hour)
    - frequency whiten

    Parameters
    ----------
    da : xr.DataArray
        should be 1D dataarray with dimension 'time'.
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz 
    tide : bool
        if true, linear/geometric time warping is applied for s1b0 peak
    tide_interp : scipy.interpolate._interpolate.interp1d
        interpolates ns timestamp to change of tide in meters
        Currently there are not catches for the fact that default is None
    '''
    # load single chunk into numpy array
    data = da.values

    # remove mean
    data_nm = data - np.nanmean(data)

    # set_nan = 0
    data_nm[np.isnan(data_nm)] = 0

    # reshape data to be segments of length W
    data_rs = np.reshape(data_nm, (int(len(data_nm)/(W*Fs)), int(W*Fs)))

    if tide:
        D = 1523
        L = 3186
        c = 1481

        sample_time_coords = da.time[::W*Fs].values.astype(int)
        tidal_shift = tide_interp(sample_time_coords)
        time_shift = 2*np.sqrt(D**2 + (L/2)**2)/c - 2*np.sqrt((D-tidal_shift)**2 + (L/2)**2)/c
        timebin_shift = np.expand_dims(time_shift/0.005, 1)
        k = np.expand_dims(np.arange(0,W*Fs + 1), 0)
        phase_shift = np.exp(-1j*2*np.pi/(W*Fs)*k*timebin_shift)
        data_shift_f = scipy.fft.fft(
            np.hstack((data_rs, np.zeros((data_rs.shape[0], 1)))), axis=1) * phase_shift
        # force shifted signal to be real
        data_shift_f[:,int(data_shift_f.shape[1]/2+1):] = np.flip(np.conjugate(data_shift_f[:,1:int(data_shift_f.shape[1]/2 + 1)]),axis=1)

        data_shift = np.real(scipy.fft.ifft(data_shift_f))[:,:-1]
    
    else:
        data_shift = data_rs


    # filter data
        # filter is 4th order butterwork [0.01, 0.9]
    b = [0.63496904 , 0, - 2.53987615,  0, 3.80981423, 0, - 2.53987615, 0, 0.63496904]
    a = [ 1, -0.73835614, -2.84105805, 1.53624064, 3.3497155, -1.14722815, -1.86018017, 0.29769033, 0.40318603]
    
    data_filt = signal.filtfilt(b,a,data_shift, axis=1)

    # clip data
    std = np.nanstd(data_filt)
    data_clip = np.clip(data_filt, -3*std, 3*std)

    # frequency whiten data
    data_whiten = freq_whiten(data_clip, b,a)

    # For now, flatten back into a timeseries
    data_flat = data_whiten.flatten()

    # reconstruct xr.DataArray
    preprocessed_da = xr.DataArray(data_flat, dims=da.dims, coords=da.coords, attrs=da.attrs)

    return preprocessed_da

def freq_whiten(data, b,a):
    '''
    freq_whiten - force magnitude of fft to be filter response magnitude
        (retains phase information)

    Parameters
    ----------
    data : np.array
        array of shape [segment, time] containing segments of time
        series data to individually whiten
    
    '''
    # window data
    win = np.expand_dims(scipy.signal.windows.hann(data.shape[1]), 1)

    data_win = (data.T * win).T

    # take fft
    data_f = scipy.fft.fft(data_win, axis=1)
    data_phase = np.angle(data_f)

    # get magnitude of filter
    #_,H = signal.freqz(b,a, data.shape[1])

    pulse = signal.unit_impulse(data.shape[1], idx='mid')
    H = np.abs(scipy.fft.fft(signal.filtfilt(b,a,pulse)))
    H = np.expand_dims(H, 1)

    # construct whitened signal
    data_whiten_f = (np.exp(data_phase * 1j).T * np.abs(H)**2).T # H is squared because of filtfilt

    data_whiten = np.real(scipy.fft.ifft(data_whiten_f, axis=1))

    return data_whiten