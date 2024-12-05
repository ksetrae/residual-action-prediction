Code implementing ideas from paper named 
**"Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction"**
(https://arxiv.org/abs/2207.09705)

# How to run
1. Install all libraries as in requirements.txt
2. run record_expert_data.py
3. run train_and_eval.ipynb

# Notes
**Note**: this code is adapted from private repo where it was based on private API, so
architecture of this API might not be optimal.  

Also because of this reason, it also wasn't tested on the same datasets as in original paper, 
and not even on the same data modality, hence model architecture is different.

It only follows main ideas of the paper.

Tested with python 3.10, pytorch 2.0

# TODO
- compare to behavioral cloning on observation history (BCOH), behavioral cloning from single observation (BCSO) as in the paper
- evaluation on MuJoCo/CARLA as in the paper