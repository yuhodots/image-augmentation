#!/bin/bash

# Experiment options
echo
echo -n "* Please select GPU ID: "
read -r num
GPU="${num}"

# Run python file
cd ../
CUDA_VISIBLE_DEVICES=${GPU} python augmix/main.py \
  --model preactresnet18 \
  --data_dir /home/miil/Datasets/FSCIL-CEC
