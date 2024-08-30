#!/bin/bash

if [[ $# -lt 1 || $# -gt 1 ]]
then
    echo "Enter Path of File as Input. Exiting"
    exit
fi

path=$1
k=0
echo "Enter Value of k: "
read k
echo "Enter Value of metric: "
metric=0
read metric

python3 ./1.py task "$path" $k $metric
echo "Complete"
