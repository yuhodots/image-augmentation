#!/bin/bash

echo
echo "This script is for CIFAR-100 experiment
      which contains only '60' classes."

# Experiment options
options=("CIFAR-100 & Manifold mixup Preactresnet18 (original repo ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet18 (paper ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet18 (fast train ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet20")

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
if [ "${option}" = "CIFAR-100 & Manifold mixup Preactresnet18 (original repo ver.)" ]; then
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
        --mixup_alpha 2.0 \
        --partial_class True \
        --partial_class_indices 60 \
        --memo original_repo_ver
elif [ "${option}" = "CIFAR-100 & Manifold mixup Preactresnet18 (paper ver.)" ]; then
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
        --schedule 1000 1500 \
        --gammas 0.1 0.1 \
        --train mixup_hidden \
        --mixup_alpha 2.0 \
        --partial_class True \
        --partial_class_indices 60 \
        --memo paper_ver
elif [ "${option}" = "CIFAR-100 & Manifold mixup Preactresnet18 (fast train ver.)" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python mixup/main.py \
        --dataset cifar100 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/mixup/ \
        --labels_per_class 500 \
        --arch preactresnet18  \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 200 \
        --schedule 100 150 \
        --gammas 0.1 0.1 \
        --train mixup_hidden \
        --mixup_alpha 2.0 \
        --partial_class True \
        --partial_class_indices 60 \
        --memo fast_train_ver
elif [ "${option}" = "CIFAR-100 & Manifold mixup Preactresnet20" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python mixup/main.py \
        --dataset cifar100 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/mixup/ \
        --labels_per_class 500 \
        --arch preactresnet20  \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 2000 \
        --schedule 1000 1500 \
        --gammas 0.1 0.1 \
        --train mixup_hidden \
        --mixup_alpha 2.0 \
        --partial_class True \
        --partial_class_indices 60
else
    echo "There is no option for '${option}'"
fi
