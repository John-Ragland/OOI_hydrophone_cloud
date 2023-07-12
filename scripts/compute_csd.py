import pandas as pd
import ODLintake
from xrsignal.xrsignal import xrsignal
from dask.distributed import Client, LocalCluster
from OOI_hydrophone_cloud import utils as hdata_utils
import dask
import xarray as xr
from NI_tools.NI_tools import calculate

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    lf_hdata = ODLintake.open_ooi_lfhydrophones()
    lf_hdata_slice = hdata_utils.slice_ds(lf_hdata, pd.Timestamp(
        '2015-01-01'), pd.Timestamp('2023-01-01'), include_coord=False)[['AXCC1', 'AXEC2']]
    time_coord = pd.date_range(pd.Timestamp(
        '2015-01-01'), pd.Timestamp('2022-12-31T23:59:59.99999'), freq='1H')
    lf_hdata_slice_pp = calculate.preprocess(lf_hdata_slice, dim='time', include_coords=False)
    csd = xrsignal.csd(lf_hdata_slice_pp, dim='time', nperseg=4096, fs=200, dB=False, average='mean')
    csd = csd.assign_coords({'time':time_coord})
    csd = csd.chunk({'time':512})
    csd.attrs = {'units':'dB rel $\mathrm{\\frac{\mu Pa^2}{Hz}}$'}

    print(csd)
    csd_ds = xr.Dataset({'csd':csd})
    fn = '/datadrive/lfhydrophone/AXCC1_AXEC2_csd_4096pts.nc'
    csd_ds.to_netcdf(fn)