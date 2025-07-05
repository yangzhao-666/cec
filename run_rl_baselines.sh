#!/usr/bin/env bash

NUM_STEPS=100000
EVAL_STEPS=5000

for LR in 0.0003 0.001
do
    for NS in 16 32 8
    do
        python sac.py --num_steps $NUM_STEPS --eval_freq $EVAL_STEPS --wandb --batch_size $NS --lr $LR &
    done
done
