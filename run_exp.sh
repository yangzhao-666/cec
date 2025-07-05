#!/usr/bin/env bash

NUM_STEPS=100000
EVAL_STEPS=5000

for TAU in 0.1 10
do
    for K in 3 5 1
    do
        for ACT_N in 0.1 
        do
            for T in 1 3
            do
                for ENV in 'Safe'
                do
                    for DT in 0.1
                    do
                        python train.py --distance_threshold $DT --eps_decay_steps $NUM_STEPS --num_steps $NUM_STEPS --eval_freq $EVAL_STEPS --wandb --act_noise $ACT_N --exploration 'random' --T $T --k $K --tau $TAU --env $ENV &
                    done
                done
            done
        done
    done
done
