#!/bin/bash - 
#===============================================================================
#
#          FILE: mnist-few-shot.sh
# 
#         USAGE: ./mnist-few-shot.sh 
# 
#   DESCRIPTION: Run the main mnist few shot experiment with mlp and spn network.
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Steven Lang 
#  ORGANIZATION: 
#       CREATED: 01/29/2019 15:01
#      REVISION:  ---
#===============================================================================

set -e

TAG="two-spn-layers-big"
# Create base directory based on the current time
base_dir="./results/fewshot"
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
  ./env/bin/python src/models/main_fewshot.py \
    --result-dir $result_dir \
    --batch-size 64 \
    --lr 0.001 \
    --net $n \
    --epochs 100 \
    --njobs 26 \
    --experiment-name $name

done
