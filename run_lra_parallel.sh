#!/bin/bash

# Run each script on a separate GPU
CUDA_VISIBLE_DEVICES=4 taskset -c 0-10 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S4D-Real model.bidirectional=True &
CUDA_VISIBLE_DEVICES=5 taskset -c 11-20 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S4D-Complex model.bidirectional=True &
CUDA_VISIBLE_DEVICES=6 taskset -c 21-30 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S6-Real model.bidirectional=True &
CUDA_VISIBLE_DEVICES=7 taskset -c 31-40 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S6-Complex model.bidirectional=True &
CUDA_VISIBLE_DEVICES=4 taskset -c 41-50 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S4D-Real model.bidirectional=False &
CUDA_VISIBLE_DEVICES=5 taskset -c 51-60 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S4D-Complex model.bidirectional=False &
CUDA_VISIBLE_DEVICES=6 taskset -c 61-70 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S6-Real model.bidirectional=False &
CUDA_VISIBLE_DEVICES=7 taskset -c 71-80 python -m ssm-benchmark-master.train --config cifar-10-mamba.yaml --overrides model.ssm_type=S6-Complex model.bidirectional=False &

wait
