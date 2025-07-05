# Continuous Episodic Control
IEEE Conference on Games 2023

[Zhao Yang](https://yangzhao-666.github.io), [Thomas Moerland](https://thomasmoerland.nl), [Mike Preuss](https://scholar.google.se/citations?user=KGlyGUcAAAAJ&hl=en), [Aske Plaat](https://askeplaat.wordpress.com)

<img src="https://github.com/yangzhao-666/cec/blob/main/CEC.png" width="600" height="300">

---
To learn more:
- [paper](https://arxiv.org/pdf/2211.15183)

If you find our paper or code useful, please reference us:
```
@inproceedings{yang2023continuous,
  title={Continuous episodic control},
  author={Yang, Zhao and Moerland, Thomas M and Preuss, Mike and Plaat, Aske},
  booktitle={2023 IEEE Conference on Games (CoG)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```
### Running Experiments
--- 
You can run the experiments by running ```python train.py --distance_threshold $DT --eps_decay_steps $NUM_STEPS --num_steps $NUM_STEPS --eval_freq $EVAL_STEPS --wandb --act_noise $ACT_N --exploration 'random' --T $T --k $K --tau $TAU --env $ENV```.

Please be noted that there are many hyper-parameters in this work and they are quite senstive. In order to fully reproduce the results presented you need to set every hyperparameter the same as ones reported in the paper.

### Code Overview
---
The structure of the code base.
```
  |- cec.py            # the implementation of CEc agent
  |- train.py          # the training logic
  |- ToyExample.ipynb  # illustration of the toy example shown in the paper
  |- utils.py          # utils functions
```
