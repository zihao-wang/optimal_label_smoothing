#!/bin/bash

for p in 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
do
    for noise_rate in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
	python3 main.py --result_dir results/grid_search/cifar100-ResNet --dataset cifar100 --noise_rate $noise_rate --smoothing $p --batch_size 128 --noise_type symmetric --n_epoch 100 --lambda_type nll
    done
done

# python3 main.py --result_dir results/grid_search --dataset mnist --noise_rate 0.0 --smoothing 1.0 --lr 0.1 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll
# python3 main.py --result_dir results/grid_search --dataset cifar10 --noise_rate 0.0 --smoothing 1.0 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll
