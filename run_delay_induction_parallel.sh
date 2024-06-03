#!/bin/bash

# Run each script on a separate GPU
CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 python train.py --config induction.yaml --overrides model.ssm_type=S4D-Real dataset.auto_regressive=True &
CUDA_VISIBLE_DEVICES=1 taskset -c 11-20 python train.py --config induction.yaml --overrides model.ssm_type=S4D-Complex dataset.auto_regressive=True &
CUDA_VISIBLE_DEVICES=2 taskset -c 21-30 python train.py --config induction.yaml --overrides model.ssm_type=S6-Real dataset.auto_regressive=True &
CUDA_VISIBLE_DEVICES=3 taskset -c 31-40 python train.py --config induction.yaml --overrides model.ssm_type=S6-Complex dataset.auto_regressive=True &

CUDA_VISIBLE_DEVICES=4 taskset -c 41-50 python train.py --config induction.yaml --overrides model.ssm_type=S4D-Real dataset.auto_regressive=True dataset.induction_len=2&
CUDA_VISIBLE_DEVICES=5 taskset -c 51-60 python train.py --config induction.yaml --overrides model.ssm_type=S4D-Complex dataset.auto_regressive=True dataset.induction_len=2&
CUDA_VISIBLE_DEVICES=6 taskset -c 61-70 python train.py --config induction.yaml --overrides model.ssm_type=S6-Real dataset.auto_regressive=True dataset.induction_len=2&
CUDA_VISIBLE_DEVICES=7 taskset -c 71-80 python train.py --config induction.yaml --overrides model.ssm_type=S6-Complex dataset.auto_regressive=True dataset.induction_len=2&


#CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Real dataset.auto_regressive=True dataset.copy_token=False &
#CUDA_VISIBLE_DEVICES=1 taskset -c 11-20 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Complex dataset.auto_regressive=True dataset.copy_token=False&
#CUDA_VISIBLE_DEVICES=2 taskset -c 21-30 python train.py --config delay.yaml --overrides model.ssm_type=S6-Real dataset.auto_regressive=True dataset.copy_token=False&
#CUDA_VISIBLE_DEVICES=3 taskset -c 31-40 python train.py --config delay.yaml --overrides model.ssm_type=S6-Complex dataset.auto_regressive=True dataset.copy_token=False&
#
#CUDA_VISIBLE_DEVICES=4 taskset -c 41-50 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Real dataset.auto_regressive=False dataset.copy_token=False&
#CUDA_VISIBLE_DEVICES=5 taskset -c 51-60 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Complex dataset.auto_regressive=False dataset.copy_token=False&
#CUDA_VISIBLE_DEVICES=6 taskset -c 61-70 python train.py --config delay.yaml --overrides model.ssm_type=S6-Real dataset.auto_regressive=False dataset.copy_token=False&
#CUDA_VISIBLE_DEVICES=7 taskset -c 71-80 python train.py --config delay.yaml --overrides model.ssm_type=S6-Complex dataset.auto_regressive=False dataset.copy_token=False&
#
#CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Real dataset.auto_regressive=True dataset.copy_token=True&
#CUDA_VISIBLE_DEVICES=1 taskset -c 11-20 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Complex dataset.auto_regressive=True dataset.copy_token=True&
#CUDA_VISIBLE_DEVICES=2 taskset -c 21-30 python train.py --config delay.yaml --overrides model.ssm_type=S6-Real dataset.auto_regressive=True dataset.copy_token=True&
#CUDA_VISIBLE_DEVICES=3 taskset -c 31-40 python train.py --config delay.yaml --overrides model.ssm_type=S6-Complex dataset.auto_regressive=True dataset.copy_token=True&
#
#CUDA_VISIBLE_DEVICES=4 taskset -c 41-50 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Real dataset.auto_regressive=False dataset.copy_token=True&
#CUDA_VISIBLE_DEVICES=5 taskset -c 51-60 python train.py --config delay.yaml --overrides model.ssm_type=S4D-Complex dataset.auto_regressive=False dataset.copy_token=True&
#CUDA_VISIBLE_DEVICES=6 taskset -c 61-70 python train.py --config delay.yaml --overrides model.ssm_type=S6-Real dataset.auto_regressive=False dataset.copy_token=True&
#CUDA_VISIBLE_DEVICES=7 taskset -c 71-80 python train.py --config delay.yaml --overrides model.ssm_type=S6-Complex dataset.auto_regressive=False dataset.copy_token=True&

wait
