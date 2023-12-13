# Scripts

## mseed2zarr.py
- take mseed files located at https://lfhydrophone.blob.core.windows.net/miniseed2 and convert to zarr
- currently, the zarr is stored on 1TB data disk, and not directly written to blob storage
    - for some reason writing directly to blob storage takes way longer.
- It successively downloads a single mseed file using azcopy and then appends to zarr store
- it's currently hard coded to the specific miniseed file structure (and file lengths)

## download_file.sh
- downloads a specific miniseed file from https://lfhydrophone.blob.core.windows.net/miniseed2
- it takes the specific file name as input
- called by mseed2zarr.py for each mseed file

## log.txt
- log of each integer index for zarr store and a timestep, representing each individual miniseed file