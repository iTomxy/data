#!/bin/bash
cd ..
/usr/local/R2018a/bin/matlab -nodesktop -nosplash -r "check_non_jpg('image_path', '/home/dataset/nuswide/Flickr')"
python replace-fake-jpg.py
