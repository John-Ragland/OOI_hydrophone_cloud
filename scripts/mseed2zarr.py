'''
mseed2zarr.py - this converts the mseed files saved in lfhydrophone/mseed2 to a zarr store

mseed files are not uniformly created, because I was simultaneously developing the code while downloading the files
This results in alot of strange handling of the timestamps around 2016

03.06.2023 - updating LF hydrophone zarr store to contain data through 2022. I'm thinking that I'm
    going to create the zarr store from scratch again and that I will still not have the ability 
    to append. I'm creating it from scratch so that I can add calibration information
'''
import obspy
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import io
from tqdm import tqdm
import xarray as xr
from dask.distributed import Client

connect_str = os.environ['AZURE_CONSTR']
# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
container_name = 'miniseed2'
# Create container
container_client = blob_service_client.get_container_client(container_name)
storage_options = {'account_name':'lfhydrophone', 'account_key':os.environ['AZURE_KEY']}

station_names = ['AXBA1','AXCC1','AXEC2','HYSB1', 'HYS14']
calib_vals = [2257.53, 2480.98, 2421.9, 2311.11, 2499.09] # counts/PA
calib = dict(zip(station_names, calib_vals))

zarr_dir = '/datadrive/ooi_lfhydrophones.zarr'

# List the blobs in the container
blob_list = list(container_client.list_blobs())
last_int_idx = -1

start_idx = 0

for k, blob in tqdm(enumerate(blob_list)):
    if k < start_idx:
        continue
    elif k == start_idx:
        # get last int idx of ds
        ds1 = xr.open_zarr(zarr_dir)
        last_int_idx = ds1['AXBA1'].time[-1]

    if k == 0:
        # create log file heading
        with open('log.txt', 'w') as f:
            f.write('int_idx, date_string\n')
    
    # reset file empty flag
    file_empty = False

    # Get expected start and endtimes for specific mseed file
    # file segment length is related to how the files were downloaded
    starttime = pd.Timestamp(blob.name[:-9].replace('_','-'))
    if starttime >= pd.Timestamp('2016-01-29'):
        endtime = starttime + pd.Timedelta(days=30) - pd.Timedelta(seconds=0.005)
    else:
        endtime = starttime + pd.Timedelta(days=1) - pd.Timedelta(seconds=0.005)
    starttime = obspy.UTCDateTime(starttime)
    endtime = obspy.UTCDateTime(endtime)

    fn = '../data/temp.miniseed'
    
    # remove file if exists
    if os.path.exists(fn):
        os.system(f'rm {fn}')

    # For some reason this is painfully slow, but azcopy works so...
    #container_client = blob_service_client.get_container_client(container=container_name)
    #print(f"Downloading blob {blob.name}...")
    #with open(fn, "wb") as download_file:
    #    try:
    #        download_file.write(
    #            container_client.download_blob(blob.name).readall())
    #    except:
    #        file_empty = True
    
    # download file using azcopy
    print(f"Downloading blob {blob.name}...")
    os.system(f"bash download_file.sh {blob['name']}")

    # calculate number of points that should be there
    npts = int((endtime - starttime)*200 + 1)

    data = {}

    if not file_empty:
        # open miniseed file and merge
        reclen = 512
        chunksize = 100000 * reclen # Around 50 MB
        trs = []

        print('openning mseed file...')
        with io.open(fn, "rb") as fh:
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

        # force sampling rate to be exactly 200
        for tr in stream:
            tr.stats.sampling_rate = 200
        print('merging stream...')
        # merge stream
        merged_stream = stream.merge()

        print('constructing dataset...')
        # Construct xarray dataset
        for tr in merged_stream:
            # convert to float 32, fill masked values with nan, convert to Pa
            if isinstance(tr.data, np.ma.core.MaskedArray):
                data_single = tr.data.astype(np.float32)
                np.ma.set_fill_value(data_single, np.nan)
                data[tr.stats.station] = data_single.filled()
            elif isinstance(tr.data, np.ndarray):
                data[tr.stats.station] = tr.data.astype(np.float32)
            else:
                raise Exception(f'type of data {type(tr.data)}. expected numpy.ndarray or numpy.ma.core.MaskedArray')
            

            if (tr.stats.starttime == starttime) & (tr.stats.endtime == endtime):
                pass
            elif (tr.stats.starttime > starttime) | (tr.stats.endtime < endtime):
                # trace doesn't cover full expected range of data
                # need to pad nan values
                if tr.stats.starttime > starttime:
                    npts_missing = round((tr.stats.starttime - starttime)*200)
                    data[tr.stats.station] =  np.concatenate((np.ones(npts_missing)*np.nan, data[tr.stats.station]))
                if tr.stats.endtime < endtime:
                    npts_missing = round((endtime - tr.stats.endtime)*200)
                    data[tr.stats.station] = np.concatenate((data[tr.stats.station], np.ones(npts_missing)*np.nan))

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

    # File is empty, fill with NaNs
    else:
        for station in station_names:
            data[station] = np.ones(npts)*np.nan

    print('save to zarr...')
    
    # Convert data from numpy to xr.DataArray
    for key in list(data.keys()):
        data[key] = xr.DataArray(data[key], dims=['time'])
    
    # Calibrate data
    for key in list(data.keys()):
        data[key] = data[key] / calib_vals[key]
        
    attrs = {
        'network':'OO',
        'channel':'HDH',
        'sampling_rate':200,
        'delta':0.005,
        'units':'Pa'
    }

    # create dataset
    ds = xr.Dataset(data, attrs=attrs)
    ds = ds.chunk({'time':3600*200})

    # Save dataset to cloud
    if k == 0:
        ds.to_zarr(zarr_dir, mode='w-')
    else:
        ds.to_zarr(zarr_dir, append_dim='time')

    # log integer idx and starttime
    with open('log.txt', 'a') as f:
        f.write(f'{int(last_int_idx+1)}, {starttime}\n')

    # get last int idx of ds
    ds1 = xr.open_zarr(zarr_dir)
    last_int_idx = ds1['AXBA1'].time[-1]
    
    # remove all variables to avoid memory overload
    try:
        del ds, data, tr, merged_stream, stream, ds1
    except NameError:
        continue
