#!/bin/bash - 
#===============================================================================
#
#          FILE: ./mnist-multilabel.sh
# 
#         USAGE: ./mnist-multilabel.sh n  # n == number of labels
# 
#   DESCRIPTION: Run the main mnist multilabel experiment with mlp and spn network.
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Steven Lang 
#  ORGANIZATION: 
#       CREATED: 04/10/2019 15:01
#      REVISION:  ---
#===============================================================================

set -e


# Hyperparameters
batch_size=128
test_batch_size=128
lr=0.0001
l2=0.000
epochs=100
njobs=4
n_labels=10
n_digits=10
canvas_size=50
seed=0
resnet_arch="resnet18"
arch="resnet"
result_dir="results"
reuse_base_dir="results/multilabel-mnist/190616_2132_gridsearch/"
cuda_device_id="1 2 3"

# Specify arguments via array to allow for comments
args=(
  --experiment "multilabel-mnist" 
  --experiment-name $arch 
  --result-dir $result_dir 
  --reuse-base-dir $reuse_base_dir 
  --batch-size $batch_size 
  --test-batch-size $test_batch_size 
  --lr $lr 
  --l2 $l2 
  --epochs $epochs 
  --njobs $njobs 
  --n-labels $n_labels 
  --n-digits $n_digits 
  --canvas-size $canvas_size 
  --seed $seed 
  --cuda 
  --cuda-device-id $cuda_device_id 
  --net $arch
)

# Run resnet
./env/bin/python src/models/main_experiment.py "${args[@]}"

  # --experiment "multilabel-mnist" \
  # --experiment-name $arch \
  # --result-dir $result_dir \
  # --reuse-base-dir $reuse_base_dir \
  # --batch-size $batch_size \
  # --test-batch-size $test_batch_size \
  # --lr $lr \
  # --l2 $l2 \
  # --epochs $epochs \
  # --njobs $njobs \
  # --n-labels $n_labels \
  # --n-digits $n_digits \
  # --canvas-size $canvas_size \
  # --seed $seed \
  # --cuda \
  # --cuda-device-id $cuda_device_id \
  # --net $arch
