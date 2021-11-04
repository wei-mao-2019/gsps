#!/bin/bash
CFG=h36m
# training
python train.py --cfg $CFG --gpu_index 0

# evaluation
python eval.py --cfg $CFG --gpu_index 1 --mode stats --nk 50 --multimodal_threshold 0.5

# evaluation with fixed lower body motion
python eval.py --cfg $CFG --gpu_index 1 --mode stats --nk 50 --multimodal_threshold 0.5 --fixlower
