#!/bin/bash
set -e

#
# Link images \& labels of RibSebv2
# Check https://github.com/M3DV/RibSeg/blob/ribsegv1/data_prepare.py for source folder structure.
#

src=$HOME/data/ribseg
dst=data/ribsegv2


echo link labels
dst_lab_p=$dst/label
if [ ! -d $dst_lab_p ]; then
    mkdir -p $dst_lab_p
fi
src_lab_p=$src/ribseg_v2/seg
for f in `ls $src_lab_p`; do
    ln -s $src_lab_p/$f $dst_lab_p/$f
done


echo link images
dst_img_p=$dst/image
if [ ! -d $dst_img_p ]; then
    mkdir -p $dst_img_p
fi
src_img_p=$src/ribfrac
for subp in \
    ribfrac-train-images-1/Part1 \
    ribfrac-train-images-2/Part2 \
    ribfrac-val-images \
    ribfrac-test-images;
do
    echo $subp
    for f in `ls $src_img_p/$subp`; do
        ln -s $src_img_p/$subp/$f $dst_img_p/$f
    done
done
