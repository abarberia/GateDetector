#!/bin/bash

usage()
{
    echo "usage: video.sh [-v video] [-w weights] [-o output]" 
    echo "\t[-v --video ]:\t Name of video file to process. Default is '../data/video.avi'."
    echo "\t[-w --weights]:\t DNN weights used. Default is 'weights/size_360.weights'."
    echo "\t[-o --output]:\t Name of output file. Default is '../data/processed_video.avi'"
}

cd darknet
directory="$(dirname $0)"
video="$directory/../data/video.avi"
weights="weights/size_360.weights"
output="$directory/../data/processed_video.avi"
while [ "$#" != 0 ]; do
	case $1 in
		-v | --video)	video=$2
				shift 2
				;;
		-w | --weights)	weights=$2
				shift 2
				;;
		-o | --output) output=$2
				shift 2
				;;
		-h | --help)	usage
				exit
				;;
		* )		usage
				exit
	esac
done

./darknet detector demo obj.data cfg/obj_test.cfg $weights $video -dont_show -out $output -thresh 0.6
