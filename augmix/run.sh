#!/bin/bash

# Experiment options
exp_opt=("CIFAR-100 |    AugMix    | preactresnet18" \
         "CIFAR-100 |    AugMix    | preactresnet20" )
partial_class_opt=("True" "False")

echo
echo "* Please select the experiment option."
PS3="number: "
select opt in "${exp_opt[@]}"; do
    EXP="${opt}"
    break
done

echo
echo "* Please select the partial class option."
PS3="number: "
select opt in "${partial_class_opt[@]}"; do
    PCB="${opt}"
    break
done

PCI=100
if [ "${PCB}" = "True" ]; then
    echo
    echo -n "* Please select partial_class_indices: "
    read -r num
    PCI="${num}"
fi

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
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
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
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
        --memo ''
else
    echo "There is no option for '${EXP}'"
fi