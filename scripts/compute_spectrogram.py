import pandas as pd
import ODLintake
from xrsignal import xrsignal
from dask.distributed import Client, LocalCluster
from OOI_hydrophone_cloud import utils as hdata_utils
import dask

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    lf_hdata = ODLintake.open_ooi_lfhydrophones()
    lf_hdata_slice = hdata_utils.slice_ds(lf_hdata, pd.Timestamp(
        '2015-01-01'), pd.Timestamp('2023-01-01'), include_coord=False)
    time_coord = pd.date_range(pd.Timestamp(
        '2015-01-01'), pd.Timestamp('2022-12-31T23:59:59.99999'), freq='1H')
    specs = xrsignal.welch(lf_hdata_slice*1e6, dim='time', nperseg=1024, fs=200, dB=True, average='median')
    specs = specs.assign_coords({'time':time_coord})
    specs = specs.chunk({'time':512})
    specs.attrs = {'units':'dB rel $\mathrm{\\frac{\mu Pa^2}{Hz}}$'}

    print(specs)

    fn = '/datadrive/lfhydrophone/lfhydrophone_spectrogram_median.zarr'
    specs.to_zarr(fn)