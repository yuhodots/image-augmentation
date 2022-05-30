#!/bin/bash

# Experiment options
echo
echo -n "* Please select GPU ID: "
read -r num
GPU="${num}"

# Run python file
cd ../
CUDA_VISIBLE_DEVICES=${GPU} python augmix/main.py \
    --dataset cifar100 \
    --data_dir /home/miil/Datasets/FSCIL-CEC \
    --result_dir results/augmix/ \
    --arch preactresnet18 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --decay 0.0001 \
    --epochs 200 \
    --schedule 100 150 \
    --gammas 0.1 0.1 \
    --memo ''
