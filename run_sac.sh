#!/usr/bin/env bash

NUM_STEPS=100000
EVAL_STEPS=5000

for env in 'Swimmer-v2' 'Walker2d-v2' 'HalfCheetah-v2' 'Hopper-v2'
do
    python train_sac.py --num_steps $NUM_STEPS --eval_freq $EVAL_STEPS --wandb --env $env &
done
