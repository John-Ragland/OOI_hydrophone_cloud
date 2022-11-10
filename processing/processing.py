'''
NCCF.py - building methods that utilizide Dask for calculate cross-correlation estimates

Pre-processing methods will hopefully be modular eventually 

'''

from scipy import signal
import scipy
import xarray as xr
import numpy as np
from numpy import matlib
from OOI_hydrophone_cloud import utils
from OOI_hydrophone_cloud.processing import processing

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

    return data_whiten

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

def psd_estimate_chunk(da, calibration, nperseg=512, noverlap=256, window='hann', average='mean', ):
    '''
    psd_estimate - estimates PSD for da along 'time' dimension
        PSD estimate:
        - uses Welch Method (scipy.signal implementation)

    Parameters
    ----------
    da : xarray.DataArray
        time series data with dimension time
    time_base : pd.Timestamp
        time coordinate at start of
    nperseg : int
        passed to signal.welch. Default (512)
    noverlap : int
        passed to signal.welch. Default (256)
    window : string
        passed to signal.welch. Default ('hann')
    average : string
        passed to signal.welch. Default ('mean')
    calibration : float
        calibration for hydrophone in counts / Pascal
    '''

    # load data
    x = da.values/calibration

    f, Pxx = signal.welch(
        x,
        fs=200,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        average=average
    )

    Pxx_db = 10*np.log10(Pxx/(1e-12))
    
    Pxx_x = xr.DataArray(np.expand_dims(Pxx_db, 0), dims=['time', 'frequency'])

    return Pxx_x

def compute_spectrogram(x, calibration, npts=512, noverlap=256, window='hann', average='mean', compute=True):
    '''
    compute_spectrogram - calculates PSD for every chunk in time series. Function in implemented using
        dask and xarray map_blocks

    Parameters
    ----------
    x : xr.DataArray
        time series data. must have single dimension ('time')
    calibration : float
        counts / Pa for specific hydrophone
    npts : int
        number of points in frequency for each PSD (Default 512)
    noverlap : int
        passed to psd_estimate_chunk. Default (256)
    window : string
        default 'hann'
    average : string
        default 'mean'
    compute : bool
        whether to pass dask task graph or computed result. (default=True)

    Returns
    -------
    spectrogram : xr.DataArray
        dataarray with dimensions ['time', 'frequency'],
        time dimension will be size of number of chunks in x.
    '''

    # create template
    hours = len(x.chunks[0])
    template = xr.DataArray(np.ones((hours,int(npts/2+1))), dims=['time', 'frequency']).chunk({'time':1, 'frequency':int(npts/2+1)})

    spectrogram = x.map_blocks(processing.psd_estimate_chunk, args=[calibration, npts, noverlap, window, average], template=template)

    if compute:
        return spectrogram.compute()
    else:
        return spectrogram

def NCCF_chunk(ds, W=30, Fs=200):
    '''
    calculate NCCF for given dataset of time-series

    this function is defined to be used with .map_blocks() to calculate the
        NCCF stack, where each NCCF is calcualted from a single chunk

    Parameters
    ----------
    ds : xr.Dataset
        dataset with two data variables each consisting of a timeseries that
        NCCF is calculated from. each DA should have dimension time
        
        ds must only have two data dimensions

    W : float
        size of single cross correlation window (in seconds)
    Fs : float
        sampling rate (Hz). Default 200
    delay_coord : np.array
        coordinates for delay dimension
    Returns
    -------
    NCCF : xr.DataArray
        data array with dimensions ['delay']
    '''
    if len(ds) != 2:
        raise Exception('dataset must only have 2 data variabless')
    node1, node2 = list(ds.keys())

    node1_pp = preprocess(ds[node1])
    node2_pp = preprocess(ds[node2])

    R_all = signal.fftconvolve(node1_pp, np.flip(node2_pp,axis=1), axes=1, mode='full')
    R = np.mean(R_all, axis=0)

    tau = np.arange(-W+(1/Fs), W, 1/Fs)

    Rx = xr.DataArray(np.expand_dims(R, 0), dims=['time','delay'])

    return Rx

def compute_NCCF_stack(ds, W=30, Fs=200, compute=True):
    '''
    compute_NCCF_stack - takes dataset containing timeseries from two locations
        and calculates an NCCF for every chunk in the time dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        dataset must have dimension time and two data variables
    W : float
        passed to NCCF_chunk
    Fs : float
        passed to NCCF_chunk
    compute : bool
        whether to return dask task map or computed NCCF stack
    '''
    if len(ds) != 2:
        raise Exception('dataset must only have 2 data variables')
    node1, node2 = list(ds.keys())

    # create template dataarray
    da_temp = xr.DataArray(np.ones((int(ds[node1].shape[0]/(3600*Fs)), 2*W*Fs-1)), dims=['time','delay'])
    da_temp = da_temp.chunk({'delay':11999, 'time':1})  

    NCCF_stack = ds.map_blocks(processing.NCCF_chunk, template=da_temp)

    if compute:
        return NCCF_stack.compute()
    else:
        return NCCF_stack