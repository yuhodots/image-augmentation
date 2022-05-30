#!/bin/bash

# Experiment options
options=("CIFAR-10 & Manifold mixup Preactresnet18" \
         "CIFAR-100 & Manifold mixup Preactresnet18")

echo
echo "* Please select the experiment option."
PS3="number: "
select opt in "${options[@]}"; do
    option="${opt}"
    break
done

echo
echo -n "* Please select GPU ID: "
read -r num
GPU="${num}"

# Run python file
if [ "${option}" = "CIFAR-10 & Manifold mixup Preactresnet18" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python mixup/main.py \
        --dataset cifar10 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/mixup/ \
        --labels_per_class 5000 \
        --arch preactresnet18  \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 2000 \
        --schedule 500 1000 1500 \
        --gammas 0.1 0.1 0.1 \
        --train mixup_hidden \
        --mixup_alpha 2.0
elif [ "${option}" = "CIFAR-100 & Manifold mixup Preactresnet18" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python mixup/main.py \
        --dataset cifar100 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/mixup/ \
        --labels_per_class 500 \
        --arch preactresnet18  \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 2000 \
        --schedule 500 1000 1500 \
        --gammas 0.1 0.1 0.1 \
        --train mixup_hidden \
        --mixup_alpha 2.0
else
    echo "There is no option for '${option}'"
fi