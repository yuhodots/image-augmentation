#!/bin/bash

# Experiment options
exp_opt=("CIFAR-100 |    AugMix    | preactresnet18" \
         "CIFAR-100 |    AugMix    | preactresnet20" )

echo
echo "* Please select the experiment option."
PS3="number: "
select opt in "${exp_opt[@]}"; do
    EXP="${opt}"
    break
done

echo
echo -n "* Please select GPU ID: "
read -r num
GPU="${num}"

# Run python file
cd ../
if [ "${EXP}" = "CIFAR-100 |    AugMix    | preactresnet18" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python augmix/main.py \
        --dataset cifar100 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/augmix/ \
        --arch preactresnet18 \
        --batch_size 128 \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 200 \
        --schedule 100 150 \
        --gammas 0.1 0.1 \
        --memo ''
elif [ "${EXP}" = "CIFAR-100 |    AugMix    | preactresnet20" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python augmix/main.py \
        --dataset cifar100 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/augmix/ \
        --arch preactresnet20 \
        --batch_size 256 \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 200 \
        --schedule 100 150 \
        --gammas 0.1 0.1 \
        --memo ''
else
    echo "There is no option for '${EXP}'"
fi