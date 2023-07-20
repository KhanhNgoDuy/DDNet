#!/bin/bash

# Define arrays for first and second arguments
args1=("d_30_64" "d_60_64")
args2=("acc_171.pth" "acc_200.pth")

# Get the length of either array
length=${#args1[@]}

# Iterate over the indices of one of the arrays
for ((i=0; i<length; i++))
do
    echo "Evaluating: ${args1[i]} ${args2[i]}"
    python offline.py "${args1[i]}" "${args2[i]}"
done

