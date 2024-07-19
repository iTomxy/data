set -e

echo -e \\tThis is NOT necessary!
echo If you want to save disk space, \
    after running ./combine_label.py, \
    you can use this script to remove those per-class annotation .nii.gz files under \`s\<ID\>/segmentations/\'.

P=~/data/totalsegmentator
if [ ! -d $P/data ]; then
    echo No such folder: $P/data
else
    for d in `ls -d $P/data/s*/`; do
        rm -rf $d/segmentations
    done
fi
