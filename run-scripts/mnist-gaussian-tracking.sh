#!/bin/bash - 
#===============================================================================
#
#          FILE: mnist-gaussian-tracking.sh
# 
#         USAGE: ./mnist-gaussian-tracking.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 04/20/2019 12:36
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

TAG=""
# Create base directory based on the current time
base_dir="./results/gaussian-tracking"
date_str=`date +"%y-%m-%d_%Hh:%Mm"`
result_dir="$base_dir/$date_str-$TAG"
mkdir -p $result_dir

echo -e "Storing experimental results in ${result_dir}"

CUDA_VISIBLE_DEVICES="-1"
nohup ./env/bin/python src/models/main_gauss_tracking.py \
  --result-dir $result_dir \
  --batch-size 256 \
  --lr 0.001 \
  --net spn \
  --epochs 100 \
  --njobs 10 \
  --n-labels 1 \
  --seed 0 \
  --experiment-name "gaussian-tracking" &
