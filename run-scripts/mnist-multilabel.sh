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

TAG="one-layer-32-20-nlabel=5-bs=$1"
# Create base directory based on the current time
base_dir="./results/multilabel"
date_str=`date +"%y-%m-%d_%Hh:%Mm"`
result_dir="$base_dir/$date_str-$TAG"
mkdir -p $result_dir

echo -e "Storing experimental results in ${result_dir}"

# Define variable arguments
declare -a net=("spn" "mlp")

# Run all experiments
for n in "${net[@]}"
do
  name="$n"
  CUDA_VISIBLE_DEVICES="-1"
  nohup ./env/bin/python src/models/main_multilabel.py \
    --result-dir $result_dir \
    --batch-size $1 \
    --lr 0.001 \
    --net $n \
    --epochs 100 \
    --njobs 4 \
    --n-labels 5 \
    --seed $1 \
    --experiment-name $name &

done
