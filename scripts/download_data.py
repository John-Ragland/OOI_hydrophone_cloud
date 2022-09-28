'''
download_data.py - create zarr data store from OOI LF hydrophone data
John Ragland; September 27, 2022
'''

import ooipy
from datetime import datetime
from datetime import timedelta
from OOI_hydrophone_cloud import utils
from tqdm import tqdm
import xarray as xr

time_base = datetime(2015,1,1)
chunk_length = timedelta(days=1)

for k in tqdm(range(141, 2800)):
    
    start_time = time_base + (k*chunk_length)
    end_time = time_base + ((k+1)*chunk_length)

    ab = ooipy.request.hydrophone_request.get_acoustic_data_LF(
        start_time, end_time, node='Axial_Base')
    cc = ooipy.request.hydrophone_request.get_acoustic_data_LF(
        start_time, end_time, node='Central_Caldera')
    ec = ooipy.request.hydrophone_request.get_acoustic_data_LF(
        start_time, end_time, node='Eastern_Caldera')
    sb = ooipy.request.hydrophone_request.get_acoustic_data_LF(
        start_time, end_time, node='Slope_Base')
    sh = ooipy.request.hydrophone_request.get_acoustic_data_LF(
        start_time, end_time, node='Southern_Hydrate')

    ab_interp, _ = utils.interpolate_time_coord(ab, start_time, end_time)
    cc_interp, _ = utils.interpolate_time_coord(cc, start_time, end_time)
    ec_interp, _ = utils.interpolate_time_coord(ec, start_time, end_time)
    sb_interp, _ = utils.interpolate_time_coord(sb, start_time, end_time)
    sh_interp, time_grid = utils.interpolate_time_coord(sh, start_time, end_time)

    attrs = {
        'sampling_rate': 200,
        'delta': 0.005,
        'timebase': start_time.strftime('%Y-%m-%d T%H:%M:%S.%f'),
        'npts': len(time_grid),
        'network': 'OO',
        'channel': 'HDH'
    }

    da_ab = xr.DataArray(ab_interp, dims=['time'], attrs=attrs)
    da_cc = xr.DataArray(cc_interp, dims=['time'], attrs=attrs)
    da_ec = xr.DataArray(ec_interp, dims=['time'], attrs=attrs)
    da_sb = xr.DataArray(sb_interp, dims=['time'], attrs=attrs)
    da_sh = xr.DataArray(sh_interp, dims=['time'], attrs=attrs)

    ds = xr.Dataset({
        'AXBA1': da_ab,
        'AXCC1': da_cc,
        'AXEC2': da_ec,
        'HYSB1': da_sb,
        'HYS14': da_sh,
    })

    # create 1 hour chunks
    ds = ds.chunk({'time':3600*200})

    account_key = '<acount_key>'
    storage_options={'account_name': 'lfhydrophone', 'account_key': account_key}

    if k == 0:
        ds.to_zarr('abfs://hydrophonedata/lf_data.zarr',
                   storage_options=storage_options)
    else:
        ds.to_zarr(
            'abfs://hydrophonedata/lf_data.zarr',
            storage_options=storage_options,
            append_dim='time'
        )
