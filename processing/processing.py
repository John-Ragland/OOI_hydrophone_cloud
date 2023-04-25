'''
NCCF.py - building methods that utilizide Dask for calculate cross-correlation estimates

Pre-processing methods will hopefully be modular eventually 

'''

from scipy import signal
import scipy
import xarray as xr
import numpy as np
from OOI_hydrophone_cloud.processing import processing
import dask

def preprocess_chunk(da, dim, b,a,W=30, Fs=200):
    '''
    preprocess_chunk - compute basic pre-processing for NCCF for 
        a DataArray of arbitrary dimension
        
    Pre-processing is fixed to be
    - divide into 30 s segments
    - filter between 1 and 90 Hz
    - clip to 3 times std for every chunk (which is also 1 hour)
    - frequency whiten

    Parameters
    ----------
    da : xr.DataArray
        DataArray of arbitrary dimensions
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    b : np.array
        numerator coefficients for filter
    a : np.array
        denominator coefficients for filter
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz
    '''

    # transpose data to put dim last
    dims = list(da.dims)
    new_dims_order = dims.copy()
    new_dims_order.remove(dim)
    new_dims_order = new_dims_order + [dim]
    da_t = da.transpose(*new_dims_order)

    # load single chunk into numpy array
    da_np = da_t.values
    shape = da_np.shape

    da_rs = np.reshape(da_np, (shape[:-1] + (int(shape[-1]/(W*200)), W*200)))
    
    # remove mean
    da_nm = da_rs - np.nanmean(da_rs, axis=-1, keepdims=True)
    
    # filtere data
    da_filt = signal.filtfilt(b,a,da_nm, axis=-1)

    # clip data
    std = np.nanstd(da_filt, axis=-1)
    da_clip = np.clip(da_filt, -3*std[...,None], 3*std[...,None])

    # frequency whiten data
    da_whiten = freq_whiten(da_clip, b,a)
    
    # create new DataArray
    da_unstack = np.reshape(da_whiten, shape)

    da_preprocess = xr.DataArray(da_unstack, dims=new_dims_order)
    da_preprocess = da_preprocess.transpose(*dims)
    da_preprocess = da_preprocess.assign_coords(da.coords)

    return da_preprocess.transpose(*dims)

def preprocess(da, dim, W=30, Fs=200):
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
        DataArray of arbitrary dimensions
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz 

    Return
    ------
    data_whiten : np.array
        pre-procesesd data
    '''
    b,a = signal.butter(4, [0.01, 0.9], btype='bandpass')
    data_pp = da.map_blocks(preprocess_chunk, kwargs=({'dim':dim, 'b':b, 'a':a, 'W':W, 'Fs':Fs}), template=da)
    return data_pp

def freq_whiten(data, b,a):
    '''
    freq_whiten - force magnitude of fft to be filter response magnitude
        (retains phase information)

    Parameters
    ----------
    data : np.array
        array of shape [... , segment, time] containing segments of time
        series data to individually whiten. can contain other dimensions prior
        to segment and time, but segment and time must be last two dimensions
    
    '''
    # window data and compute unit pulse
    win = signal.windows.hann(data.shape[-1])
    pulse = signal.unit_impulse(data.shape[-1], idx='mid')
    for k in range(len(data.shape)-1):
        win = np.expand_dims(win, 0)
        pulse = np.expand_dims(pulse, 0)

    data_win = data * win

    # take fft
    data_f = scipy.fft.fft(data_win, axis=-1)
    data_phase = np.angle(data_f)

    H = np.abs(scipy.fft.fft(signal.filtfilt(b,a,pulse, axis=-1)))

    # construct whitened signal
    data_whiten_f = (np.exp(data_phase * 1j) * np.abs(H)**2) # H is squared because of filtfilt

    data_whiten = np.real(scipy.fft.ifft(data_whiten_f, axis=-1))

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

def NCCF_chunk(ds, stack=True):
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
    delay_coord : np.array
        coordinates for delay dimension
    stack : bool
        whether or not to return all small time correlations or stack then (default True)

    Returns
    -------
    NCCF : xr.DataArray
        data array with dimensions ['delay']
        - if stack is False, data will have dimensions ['delay', 'time']
    '''
    if len(ds) != 2:
        raise Exception('dataset must only have 2 data variabless')
    node1, node2 = list(ds.keys())

    node1_pp = preprocess(ds[node1])
    node2_pp = preprocess(ds[node2])

    print(stack)
    
    R_all = signal.fftconvolve(node1_pp, np.flip(node2_pp,axis=1), axes=1, mode='full')
    if stack:
        R = np.mean(R_all, axis=0)
        #tau = np.arange(-W+(1/Fs), W, 1/Fs)

        Rx = xr.DataArray(np.expand_dims(R, 0), dims=['time','delay'])
        return Rx
    
    else:
        Rallx = xr.DataArray(R_all, dims=['time', 'delay'])
        return Rallx

def compute_NCCF_stack(ds, W=30, Fs=200, compute=True, stack=True):
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
    stack : bool
        if true, then NCCF is stacked across chunks.
        if false, full NCCF stack is return (no averaging is done)
    '''
    if len(ds) != 2:
        raise Exception('dataset must only have 2 data variables')
    node1, node2 = list(ds.keys())

    chunk_size = ds.chunks['time'][0]
    # chunk sizes have to be the same for both data variables (i think this is required in xarray too)

    # create template dataarray
    # if stack is true, then linear stacking is computing for chunksize of ds
    if stack == True:
        dask_temp = dask.array.random.random(
            (int(ds[node1].shape[0]/chunk_size), 2*W*Fs-1))
        da_temp = xr.DataArray(dask_temp, dims=['time','delay'])
        da_temp = da_temp.chunk({'delay':int(2*W*Fs-1), 'time':1})  # single value chunks in long time
        #da_temp = xr.DataArray(np.ones((int(ds[node1].shape[0]/chunk_size), 2*W*Fs-1)), dims=['time','delay'])

    
    else:
        dask_temp = dask.array.random.random((int(ds[node1].shape[0]/(W*Fs)), 2*W*Fs-1))
        da_temp = xr.DataArray(dask_temp, dims=['time', 'delay'])
        da_temp = da_temp.chunk({'delay':int(2*W*Fs-1), 'time':int(chunk_size/Fs/W)})  #1 hour chunks in long time
        #da_temp = xr.DataArray(np.ones((int(ds[node1].shape[0]/(W*Fs)), 2*W*Fs-1)), dims=['time','delay'])

    #return processing.NCCF_chunk(ds, stack=False)
    NCCF_stack = ds.map_blocks(processing.NCCF_chunk, template=da_temp, args=[stack])
    NCCF_stack = NCCF_stack.assign_coords({'delay': np.arange(-W+1/Fs, W, 1/Fs)})
    if compute:
        return NCCF_stack.compute()
    else:
        return NCCF_stack

def compute_MultiElement_NCCF_chunk(da, time_dim='time', W=30, Fs=200):
    '''
    compute_MultiElement_NCCF - takes dataset containing timeseries from two locations
        and calculates an NCCF for each element with the first element
    
    Not completely sure if this will cause downstream problems, but this function is seperated 
        from preprocessing
    
    Parameters
    ----------
    da : xr.DataArray
        dataarray with dimensions ['time', 'element']. second dimension does not have to be named element
    time_dim : str
        name of delay dimension
    W : float
        size of window in seconds
    Fs : float
        sampling rate in Hz
    '''

    # move delay dimension to last dimension
    dims = list(da.dims)
    new_dims_order = dims.copy()
    new_dims_order.remove(time_dim)
    new_dims_order = new_dims_order + [time_dim]
    da_t = da.transpose(*new_dims_order)

    # load single chunk into numpy array
    da_np = da_t.values
    shape = da_np.shape

    # reshape into seperate segments of length W
    da_rs = np.reshape(da_np, (shape[:-1] + (int(shape[-1]/(W*Fs)), W*Fs)))
    
    # loop through all elements and compute NCCF
    for k in range(da_rs.shape[0]):
        R_all_single = np.expand_dims(signal.fftconvolve(da_rs[0,:,:], np.flip(da_rs[k,:,:],axis=-1), axes=-1, mode='full'), axis=0)
        if k == 0:
            R_all = R_all_single
        
        else:
            R_all = np.concatenate((R_all, R_all_single), axis=0)
    
    NCCF_chunk_unstacked = xr.DataArray(R_all, dims=[new_dims_order[0], 'samples', 'delay'])
    NCCF_chunk = NCCF_chunk_unstacked.mean(dim='samples')
    NCCF_chunk = NCCF_chunk.expand_dims(f'{time_dim}_chunk').transpose(new_dims_order[0], 'time_chunk', 'delay')

    print(NCCF_chunk.shape)
    return NCCF_chunk

def compute_MultiElement_NCCF(da, time_dim='time', W=30, Fs=200):
    '''
    compute_MultiElement_NCCF - takes dataarray containing timeseries with multiple elements
        and calculates an NCCF between each element and first element

    can only handle a single chunk in the element / distance dimension
    Parameters
    ----------
    da : xr.DataArray
        dataarray with dimensions ['time', 'element']. second dimension does not have to be named element
    time_dim : str
        name of time dimension
    W : float
        size of window in seconds
    Fs : float
        sampling rate in Hz
    '''

    time_idx = da.dims.index(time_dim)

    # move delay dimension to last dimension
    dims = list(da.dims)
    new_dims_order = dims.copy()
    new_dims_order.remove(time_dim)
    new_dims_order = new_dims_order + [time_dim]
    da_t = da.transpose(*new_dims_order)

    n_chunks = len(da.chunks[time_idx])

    dask_temp = dask.array.random.random((da_t.shape[0], n_chunks, 2*W*Fs-1))
    da_temp = xr.DataArray(dask_temp, dims=[new_dims_order[0], f'{time_dim}_chunk', 'delay'], name='multi-element NCCF')
    da_temp = da_temp.chunk({f'{time_dim}_chunk':1})  # single value chunks in long time

    NCCF_me = da.map_blocks(compute_MultiElement_NCCF_chunk, template=da_temp, kwargs={'time_dim':time_dim, 'W':W, 'Fs':Fs})
    return NCCF_me

def preprocess_archive(da, W=30, Fs=200, tide=False, tide_interp=None):
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

    Return
    ------
    data_whiten : np.array
        pre-procesesd data
    '''
    # load single chunk into numpy array
    data = da.values

    # remove mean
    data_nm = data - np.nanmean(data)

    # reshape data to be segments of length W
    data_rs = np.reshape(data_nm, (int(len(data_nm)/(W*Fs)), int(W*Fs)))

    # set_nan = 0
    #nan_mask = np.isnan(data_rs)
    #data_rs[nan_mask] = 0

    if tide:
        D = 1523
        L = 3186
        c = 1481

        sample_time_coords = da.time[::W*Fs].values.astype(int)
        tidal_shift = tide_interp(sample_time_coords)
        time_shift = 2*np.sqrt(D**2 + (L/2)**2)/c - 2 * \
            np.sqrt((D-tidal_shift)**2 + (L/2)**2)/c
        timebin_shift = np.expand_dims(time_shift/0.005, 1)
        k = np.expand_dims(np.arange(0, W*Fs + 1), 0)
        phase_shift = np.exp(-1j*2*np.pi/(W*Fs)*k*timebin_shift)
        data_shift_f = scipy.fft.fft(
            np.hstack((data_rs, np.zeros((data_rs.shape[0], 1)))), axis=1) * phase_shift
        # force shifted signal to be real
        data_shift_f[:, int(data_shift_f.shape[1]/2+1):] = np.flip(
            np.conjugate(data_shift_f[:, 1:int(data_shift_f.shape[1]/2 + 1)]), axis=1)

        data_shift = np.real(scipy.fft.ifft(data_shift_f))[:, :-1]

    else:
        data_shift = data_rs

    # filter data
        # filter is 4th order butterwork [0.01, 0.9]
    b = [0.63496904, 0, - 2.53987615,  0,
         3.80981423, 0, - 2.53987615, 0, 0.63496904]
    a = [1, -0.73835614, -2.84105805, 1.53624064, 3.3497155, -
         1.14722815, -1.86018017, 0.29769033, 0.40318603]

    data_filt = signal.filtfilt(b, a, data_shift, axis=1)

    # clip data
    std = np.nanstd(data_filt)
    data_clip = np.clip(data_filt, -3*std, 3*std)

    # frequency whiten data
    data_whiten = freq_whiten(data_clip, b, a)

    return data_whiten