#!/usr/bin/env bash
python ./defenses/defense_all.py \
        --data 'cifar10' --batch_size 256 --workers 16 \
        --model darts --init_channels 48 --layers 10 --model_init he_fout --data_path_cifar '/data/CIFAR' \
        --model_name "ABANDIT_CIFAR_1"  --save 'defense' --resume './weights/ABANDIT_CIFAR_1.pth.tar' \
        --print_freq 100 --gpu '1' --bn_momentum 0.1 --bn_eps 1e-3 --auxiliary  \
        --eps 0.031 --nb_iter 10 --eps_iter 0.0078
