'''
tools for creating zarr store of low frequency hydrophone data
'''

import pandas as pd
from datetime import datetime
import ooipy
import numpy as np
from scipy.interpolate import interp1d
import io
import obspy
import xarray as xr
import warnings

def interpolate_time_coord(hdata, starttime, endtime):
    '''
    interpolate_time_coord - take hydrophone data downloaded using OOIPy and 
        interpolate the time cooridinate to have a sample rate of exactly 200 Hz
        
    - interpolation is done with cubic spline, and output is stored as float32
    - if hydrophone data does not fully span specified date range, then fill values of 
        np.nan are used
    - if hdata is None, then nan values are returned for every time_grid point
        
    Parameters
    ----------
    hdata : ooipy.hydrophone.basic.HydrophoneData
        hydrophone data that has been downloaded
    starttime : datetime.datetime
        start time used to pull hydrophone data from IRIS server
    endtime : datetime.datetime
        end time used to pull hydrophone data from IRIS server
    
    Returns
    -------
    hdata_interp : np.array
        hydrophone data interpolated to uniform grid
    time_grid : np.array
        time_grid with exact sampling rate of 200 Hz. Formed from
        start time and end time variables
    '''
    time_grid = np.arange(
        pd.Timestamp(starttime).value,
        pd.Timestamp(endtime).value,
        1/200*1e9
    )

    if hdata is not None:
        # single nanosecond added to endtime to help with rounding errors
        hdata_timestamps = np.arange(
            hdata.stats.starttime.ns,
            hdata.stats.endtime.ns + 1,
            hdata.stats.delta*1e9
        )
            
        # interpolate hydrophone data
        f = interp1d(hdata_timestamps, hdata.data, kind='cubic', bounds_error=False, fill_value=np.nan)
        hdata_interp = f(time_grid).astype(np.float32)
    else:
        hdata_interp = (np.ones(time_grid.shape)*np.nan).astype(np.float32)
        
    return hdata_interp, time_grid
    
def mseed2xarray(mseed_fn, starttime, endtime):
    '''
    mseed2zarr - takes a miniseed file, downloaded form blob storage, and
        converts it to xarray dataset
    
    Parameters
    ----------
    mseed_fn : str
        file path to mseed file that will be converted to xarray
    starttime : obspy.UTCTimeStamp
        starttime for mseed file
    endtime : obspy.UTCTimeStamp
        endtime for mseed file
    '''
    # calculate number of points that should be there
    npts = int((endtime - starttime)*200 + 1)

    station_names = ['AXBA1','AXCC1','AXEC2','HYSB1', 'HYS14']
    data = {}

    # open miniseed file and merge
    reclen = 512
    chunksize = 100000 * reclen # Around 50 MB
    trs = []

    print('openning mseed file...')
    with io.open(mseed_fn, "rb") as fh:
        while True:
            with io.BytesIO() as buf:
                c = fh.read(chunksize)
                if not c:
                    break
                buf.write(c)
                buf.seek(0, 0)
                st = obspy.read(buf)
            
            for tr in st:
                trs.append(tr)

    stream = obspy.core.stream.Stream(trs)
    return stream
    # force sampling rate to be exactly 200
    for tr in stream:
        tr.stats.sampling_rate = 200
    print('merging stream...')
    # merge stream
    merged_stream = stream.merge()
    return merged_stream

    # Construct xarray dataset
    for tr in merged_stream:
        # convert to float 32 and fill masked values with nan
        data[tr.stats.station] = tr.data.astype(np.float32)
        np.ma.set_fill_value(data[tr.stats.station], np.nan)
        print('appending NaN...')
        # check starttime and endtime
        if (tr.stats.starttime == starttime) & (tr.stats.endtime == endtime):
            pass
        elif (tr.stats.starttime > starttime) | (tr.stats.endtime < endtime):
            # trace doesn't cover full expected range of data
            # need to pad nan values
            if tr.stats.starttime > starttime:
                #print(data[tr.stats.station].data.shape)
                npts_missing = round((tr.stats.starttime - starttime)*200)
                data[tr.stats.station] =  np.concatenate((np.ones(npts_missing)*np.nan, data[tr.stats.station]))
                #print('start',npts_missing, data[tr.stats.station].shape)
            if tr.stats.endtime < endtime:
                #print(data[tr.stats.station].data.shape)
                npts_missing = int((endtime - tr.stats.endtime)*200)
                data[tr.stats.station] = np.concatenate((data[tr.stats.station], np.ones(npts_missing)*np.nan))
                #print('end',npts_missing, data[tr.stats.station].shape)

            # Remove extra tick if endtime is 1 sample too long
            if (tr.stats.endtime == endtime + 0.005):
                data[tr.stats.station] = data[tr.stats.station][:-1]
        elif (tr.stats.endtime == endtime + 0.005):
            data[tr.stats.station] = data[tr.stats.station][:-1]
        else:
            raise Exception(f'current trace: {tr} not caught by starttime endtime cases')
        # check that data is the correct size before creating xr.Dataset
        if len(data[tr.stats.station]) != npts:
            print(npts)
            raise Exception(f'appending nan failed to give correct size for trace {tr}')

    # Add NaNs for any missing stations
    for station in station_names:
        try:
            data[station]
        except KeyError:
            data[station] = np.ones(npts)*np.NaN


    # Convert data from numpy to xr.DataArray
    for key in list(data.keys()):
        data[key] = xr.DataArray(data[key], dims=['time'])

    attrs = {
        'network':'OO',
        'channel':'HDH',
        'sampling_rate':200,
        'delta':0.005,
    }
    # create dataset
    ds = xr.Dataset(data, attrs=attrs)
    ds = ds.chunk({'time':3600*200})

    return ds

def int_idx(start_date, end_date):
    '''
    int_idx - get integer indices for zarr store given date bounds

    Parameters
    ----------
    start_date : pd.Timestamp
        start date for integer slice
    end_date : pd.Timestamp
        end date for integer slice

    Returns
    -------
    idx_slice : slice
        slice between start_idx and end_idx
    '''

    # this is the first time stamp of the zarr store
    time_base = pd.Timestamp('2015-01-01')

    start_idx = int((start_date - time_base).value/1e9*200)
    end_idx = int((end_date - time_base).value/1e9*200)

    return slice(start_idx,end_idx)

def date2int(date):
    time_base = pd.Timestamp('2015-01-01')
    idx = int((date - time_base).value/1e9*200)
    return idx

def slice_ds(ds, start_time, end_time, include_coord=True):
    '''
    slice_ds - slices dataset using time slice and assigns coordinates to time dimension

    Parameters
    ----------
    ds : xarray.dataset
        dataset to slice
    start_time : pd.Timestamp
        start time for slice
    end_time : pd.Timestamp
        end time for slice
    include_coord : bool
        whether or not to include time coordinate or not (Default is True)
        - when True, best use would be for slice to be no larger than one month
    '''


    ds_sliced = ds.isel({'time':int_idx(start_time, end_time)})
    if include_coord:
        if (end_time - start_time) > pd.Timedelta(31, 'd'):
            warnings.warn('slice is longer than 1 month, include_coord=True might cause memory issues')
        time_coord = pd.to_datetime(np.arange(start_time.value, end_time.value, int(5e6)))
        ds_sliced = ds_sliced.assign_coords({'time':time_coord})

    return ds_sliced