#!/bin/bash

echo download \& extract

wget -c https://zenodo.org/api/records/10927452/files-archive -O 10927452.zip

unzip 10927452.zip

for f in `ls PENGWIN_CT_train_*.zip`; do
    d=${f%%.*}
    mkdir $d
    unzip $f -d $d
done
