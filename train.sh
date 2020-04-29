#!/bin/bash


#!/bin/bash

usage()
{
    echo "usage: train.sh [-w weights] | [-h help]"
    echo "\t[-w --weights]:\t DNN weights used. Default is 'weights/size_360.weights'."
}
cd darknet
weights="weights/size_360.weights"
while [ "$#" != 0 ]; do
        case $1 in
                -w | --weights) weights=$2
                                shift 2
                                ;;
                -h | --help)    usage
                                exit
                                ;;
                * )             usage
                                exit
        esac
done

./darknet detector train obj.data cfg/obj_train.cfg $weigths 
