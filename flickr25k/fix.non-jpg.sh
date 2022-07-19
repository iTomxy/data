#!/bin/bash
cd ..
/usr/local/R2018a/bin/matlab -nodesktop -nosplash -r "check_non_jpg('image_path', '/home/dataset/flickr25k/mirflickr')"
python replace-fake-jpg.py
