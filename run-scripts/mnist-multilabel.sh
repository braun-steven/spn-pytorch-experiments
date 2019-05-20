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
batch_size=256
test_batch_size=256
lr=0.0001
epochs=100
njobs=4
n_labels=10
n_digits=10
canvas_size=50
seed=0
resnet_arch="resnet18"

echo "ARCH=$2"
# Create base directory based on the current time
base_dir="./results/resnet18"
date_str=`date +"%y-%m-%d_%Hh:%Mm"`
TAG="first-run"
result_dir="$base_dir/$date_str-$TAG"
mkdir -p $result_dir


# Define variable arguments
declare -a net=("resnet+spn-maxout" "resnet+spn")
arch=$2

# Run all experiments
# for arch in "${net[@]}"
# do
./env/bin/python src/models/main_multilabel.py \
  --result-dir $result_dir \
  --batch-size $batch_size \
  --test-batch-size $test_batch_size \
  --lr $lr \
  --epochs $epochs \
  --njobs $njobs \
  --n-labels $n_labels \
  --n-digits $n_digits \
  --canvas-size $canvas_size \
  --seed $seed \
  --experiment-name $arch \
  --cuda \
  --cuda-device-id $1 \
  --net $arch
# done



# Iterate over number of labels & digits (2..10)
# for i in `seq 6 10`
# do
#   echo -e "Storing experimental results in ${result_dir}"
#   n_digits=$i
#   n_labels=$i

#   CUDA_VISIBLE_DEVICES="$2"
#   # Create base directory based on the current time
#   base_dir="./results/multilabel-2-10-labels"
#   date_str=`date +"%y-%m-%d_%Hh:%Mm"`
#   # result_dir="$base_dir/$date_str-$TAG"
#   TAG="n-digits=$n_digits-nlabels=$n_labels"
#   result_dir="$base_dir/$TAG"
#   mkdir -p $result_dir

#   arch=$1
#   ./env/bin/python src/models/main_multilabel.py \
#     --result-dir $result_dir \
#     --batch-size $batch_size \
#     --test-batch-size $test_batch_size \
#     --lr $lr \
#     --epochs $epochs \
#     --njobs $njobs \
#     --n-labels $n_labels \
#     --n-digits $n_digits \
#     --canvas-size $canvas_size \
#     --seed $seed \
#     --experiment-name $arch \
#     --resnet-arch $resnet_arch \
#     --cuda \
#     --cuda-device-id $CUDA_VISIBLE_DEVICES \
#     --net $arch
# done
