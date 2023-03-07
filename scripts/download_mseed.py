'''
download_mseed.py - download mseed to disk and then upload to blob storage
TODO add automated refresh ability that can update the zarr store with new data

John Ragland; September 30, 2022
'''

from datetime import datetime, timedelta
from OOI_hydrophone_cloud import utils
from tqdm import tqdm
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# set up all the azure stuff

connect_str = os.environ['AZURE_CONSTR_lfhydrophone']
# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
container_name = 'miniseed2'

# Create the container
#container_client = blob_service_client.create_container(container_name)
container_client = blob_service_client.get_container_client(container_name)

# Set start date and individual file size.
time_base = datetime(2022,9,24)
chunk_length = timedelta(days=30) # size of one mseed file to be downloaded
n_chunks = 4

# if true, start and end times of each file will be printed
dry_run = False

for k in tqdm(range(0, n_chunks), position=0):
    
    starttime = time_base + (k*chunk_length)
    endtime = time_base + ((k+1)*chunk_length)

    if dry_run:
        print(starttime, endtime)
        continue

    # re-write waveform request
    lines = []
    locations = ['AXBA1', 'AXCC1', 'AXEC2', 'HYSB1', 'HYS14']
    time_string = f"{starttime.strftime('%Y-%m-%dT%H:%M:%S.%f')} {endtime.strftime('%Y-%m-%dT%H:%M:%S.%f')}"
    for n in range(5):
        lines.append(f'OO {locations[n]} -- HDH {time_string}\n')
    with open('tmp/waveform.request', 'w') as f:
        f.writelines(lines)
    f.close()

    # download day of data
    fn = f"{starttime.strftime('%Y_%m_%d')}.miniseed"
    os.system(f'wget --post-file=tmp/waveform.request -O tmp/{fn} http://service.iris.edu/fdsnws/dataselect/1/query')

    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=fn)

    # Upload the created file
    with open(f'tmp/{fn}', "rb") as data:
        blob_client.upload_blob(data)

    # remove file
    os.system(f'rm tmp/{fn}')