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

