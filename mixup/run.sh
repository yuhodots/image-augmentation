#!/bin/bash

# Experiment options
exp_opt=("CIFAR-100 & Manifold mixup Preactresnet18 (original repo ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet18 (paper ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet18 (fast train ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet20 (paper ver.)" \
         "CIFAR-100 & Manifold mixup Preactresnet20 (fast train ver.)")
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
if [ "${EXP}" = "CIFAR-100 & Manifold mixup Preactresnet18 (original repo ver.)" ]; then
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
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
        --memo original_repo_ver
elif [ "${EXP}" = "CIFAR-100 & Manifold mixup Preactresnet18 (paper ver.)" ]; then
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
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
        --memo paper_ver
elif [ "${EXP}" = "CIFAR-100 & Manifold mixup Preactresnet18 (fast train ver.)" ]; then
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
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
        --memo fast_train_ver
elif [ "${EXP}" = "CIFAR-100 & Manifold mixup Preactresnet20 (paper ver.)" ]; then
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
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
        --memo paper_ver
elif [ "${EXP}" = "CIFAR-100 & Manifold mixup Preactresnet20 (fast train ver.)" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python mixup/main.py \
        --dataset cifar100 \
        --data_dir /home/miil/Datasets/FSCIL-CEC \
        --result_dir results/mixup/ \
        --labels_per_class 500 \
        --arch preactresnet20  \
        --learning_rate 0.1 \
        --momentum 0.9 \
        --decay 0.0001 \
        --epochs 200 \
        --schedule 100 150 \
        --gammas 0.1 0.1 \
        --train mixup_hidden \
        --mixup_alpha 2.0 \
        --partial_class ${PCB} \
        --partial_class_indices ${PCI} \
        --memo fast_train_ver
else
    echo "There is no option for '${EXP}'"
fi
