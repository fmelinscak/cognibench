#!/usr/bin/env bash

mkdir -p data

ds_ids=("3601251" "1288494" "1168494" "1443332" "1443324" "3555306" "3460921")
ds_names=("fss6b" "li" "pubfe" "sc4b" "vc7b" "fer02" "doxmem2")

for i in {0..6}; do
    id=${ds_ids[$i]}
    name=${ds_names[$i]}
    path=data/${name}
    mkdir -p $path

    if [ ! -f ${path}/Data.zip ]; then
        echo "Downloading ${name}..."
        wget https://zenodo.org/record/${id}/files/Data.zip?download=1 -O ${path}/Data.zip
    fi

    echo "Unzipping ${name}..."
    cd $path
    unzip -o Data.zip
    if [ "$name" != "li" ]; then
        mv Data/* . && rmdir Data
    fi

    echo "Cleaning ${name}..."
    rm -rf *wdq*.mat
    rm -rf *SetUS*.mat
    rm Data.zip

    cd -
done
