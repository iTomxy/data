echo See: https://github.com/wasserth/TotalSegmentator
wget -O Totalsegmentator_dataset_v201.zip -c https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1

echo extract
mkdir -p totalsegmentator/data
unzip Totalsegmentator_dataset_v201.zip -d totalsegmentator/data
mv Totalsegmentator_dataset_v201.zip totalsegmentator
mv totalsegmentator/data/meta.csv totalsegmentator
