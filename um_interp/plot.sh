#!/bin/bash

# Check if a file name is provided
if [ $# -eq 0 ]; then
    echo "Please provide a file name and depth as an argument."
    echo "Usage: $0 <filename> <n_layer>"
    exit 1
fi

# Store the file name in a variable
FILE_NAME="$1"
N_LAYER=$2
N_HEAD=$3
STEP=$4
p=$5

# Run the Python script with the provided file name and other arguments
python "$FILE_NAME" --n_layer $N_LAYER --batch_size 256 --n_embd 768 --n_head $N_HEAD --lr_trgt 0.0001 --save_sequences False --warmup_steps 2048 --num_steps 100000 --num_examples 1 --num_as 128 --num_cs 128 --p_eval $p --num_ps 128 --weight_decay 1 --period_min 0 --period_max $p --act_name='relu' --step $STEP --shifts=0