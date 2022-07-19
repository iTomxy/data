#!/bin/bash
cd ..
for split in val train; do
    echo $split
    /usr/local/R2018a/bin/matlab -nodesktop -nosplash -r "check_non_jpg('image_path', '/home/dataset/COCO/${split}2017')"
    python replace-fake-jpg.py
done
