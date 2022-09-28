'''
tools for creating zarr store of low frequency hydrophone data
'''

import pandas as pd
from datetime import datetime
import ooipy
import numpy as np
from scipy.interpolate import interp1d

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
    
    