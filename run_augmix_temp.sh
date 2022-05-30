#!/bin/bash

echo
echo -n "* Please select GPU ID: "
read -r num
GPU="${num}"

CUDA_VISIBLE_DEVICES=${GPU} python augmix/main.py \
  --data_dir /home/miil/Datasets/FSCIL-CEC
