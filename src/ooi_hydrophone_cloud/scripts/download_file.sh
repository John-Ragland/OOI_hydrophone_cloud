# downloads a single miniseed file to ../data/temp.miniseed
# file to download is specifed as input

export AZCOPY_CRED_TYPE=Anonymous;
export AZCOPY_CONCURRENCY_VALUE=32;
# AZURE SAS was created by copying azcopy command from azure storage explorer
# there is definitely a better way, but I don't know it.

#echo "https://lfhydrophone.blob.core.windows.net/miniseed2/$1$AZURE_SAS_ooidata"

azcopy copy "https://ooidata.blob.core.windows.net/lfhydrophonemseed/$1$AZURE_SAS_ooidata" "temp.miniseed" --overwrite=prompt --check-md5 FailIfDifferent --from-to=BlobLocal --recursive --log-level=INFO;
unset AZCOPY_CRED_TYPE;
unset AZCOPY_CONCURRENCY_VALUE;