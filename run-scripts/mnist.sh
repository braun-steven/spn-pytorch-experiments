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

set -o nounset                              # Treat unset variables as an error
declare -a net=("mlp" "spn")

mkdir logs
  for n in "${net[@]}"
  do
    nohup ./env/bin/python src/models/main_mnist.py --batch-size 64 --lr 0.001 --result-dir ./results/result-mnist --net $n --epochs 150 --no-cuda &
  done
