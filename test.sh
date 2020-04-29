#!/bin/bash

usage()
{
    echo "usage: test.sh [-w weights] | [-h help]" 
    echo "\t[-w --weights]:\t DNN weights used. Default is 'weights/size_360.weights'."
}
cd darknet
file="predictions.txt"
weights="weights/size_360.weights"
while [ "$#" != 0 ]; do
	case $1 in
		-w | --weights)	weights=$2
				shift 2
				;;
		-h | --help)	usage
				exit
				;;
		* )		usage
				exit
	esac
done

echo $file
echo $weights
while IFS= read line
do
	set $line
	image=$1
	output=$2
	./darknet detector test obj.data cfg/obj_test.cfg $weights $image -out $output -thresh 0.001
done <"$file"
