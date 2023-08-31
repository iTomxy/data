#!/bin/bash

while read -r line; do
	f=${line#*files=}
	# echo $line, $f
	wget -c -O $f -o /dev/stdout "$line"
	# break
done < MedShapeNetDataset.txt
