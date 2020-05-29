#!/usr/bin/env bash
python ./defenses/defense_all.py \
        --data 'mnist' --batch_size 1024 --workers 16 \
        --model darts --init_channels 16 --layers 6 --model_init he_fout \
        --model_name "ABANDIT_MINIST_NORMAL_MANNUAL_1"  --save 'defense' --resume './weights/ABANDIT_MINIST_NORMAL_MANNUAL_1.pth.tar' \
        --print_freq 100 --gpu '0' --bn_momentum 0.1 --bn_eps 1e-3 \
        --eps 0.3 --nb_iter 10 --eps_iter 0.01 \
        --data_path_mnist ~/data/mnist