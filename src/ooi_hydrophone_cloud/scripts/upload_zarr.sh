export AZCOPY_CRED_TYPE=Anonymous;
export AZCOPY_CONCURRENCY_VALUE=AUTO;
azcopy copy "/datadrive/lf_hydrophone_data_test.zarr" "https://lfhydrophone.blob.core.windows.net/hydrophonedata/?sv=2021-08-06&se=2022-11-26T17%3A29%3A28Z&sr=c&sp=rwl&sig=LErsH2it%2Frv6xPoa0bafzXGULGqrDzscNFfld3%2FSt5Q%3D" --overwrite=prompt --from-to=LocalBlob --blob-type Detect --follow-symlinks --put-md5 --follow-symlinks --disable-auto-decoding=false --recursive --log-level=INFO;
unset AZCOPY_CRED_TYPE;
unset AZCOPY_CONCURRENCY_VALUE;
