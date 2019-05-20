#!/bin/bash - 
#===============================================================================
#
#          FILE: mnist.sh
# 
#         USAGE: ./mnist.sh 
# 
#   DESCRIPTION: Run the main mnist experiment with mlp and spn network.
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

# Create base directory based on the current time
base_dir="./results/cuda"
date_str=`date +"%y-%m-%d_%Hh:%Mm"`
result_dir="$base_dir/$date_str"
mkdir -p $result_dir

echo -e "Storing experimental results in ${result_dir}"

# Define variable arguments
declare -a net=("mlp" "spn")

# Run all experiments
for n in "${net[@]}"
do
  name="$n"
  nohup ./env/bin/python src/models/main_mnist.py \
    --result-dir $result_dir \
    --batch-size 1024 \
    --lr 0.001 \
    --net $n \
    --epochs 100 \
    --experiment-name $name \
    --cuda-device-id 1 \
    --cuda &
done
