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
batch_size=512
test_batch_size=512
lr=0.001
epochs=100
njobs=4
seed=0
resnet_arch="resnet18"

# Cuda device list
export CUDA_VISIBLE_DEVICES="0"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# Create base directory based on the current time
base_dir="./results/cifar100"
date_str=`date +"%y-%m-%d_%Hh:%Mm"`
TAG="first-run"
result_dir="$base_dir/$date_str-$TAG"
mkdir -p $result_dir


# Define variable arguments
declare -a net=("resnet" "resnet+spn")

# Run all experiments
for arch in "${net[@]}"
do
./env/bin/python src/models/main_cifar100.py \
  --result-dir $result_dir \
  --batch-size $batch_size \
  --test-batch-size $test_batch_size \
  --lr $lr \
  --epochs $epochs \
  --njobs $njobs \
  --seed $seed \
  --experiment-name $arch \
  --cuda \
  --cuda-device-id $CUDA_VISIBLE_DEVICES \
  --net $arch
done
