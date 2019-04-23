
set -e

TAG="spn-struct-n-gaussians"
# Create base directory based on the current time
base_dir="./results/n-gaussians"
date_str=`date +"%y-%m-%d_%Hh:%Mm"`
result_dir="$base_dir/$date_str-$TAG"
mkdir -p $result_dir

echo -e "Storing experimental results in ${result_dir}"


# Run all experiments
for n in `seq 2 10`
do
  name="n-gaussians=$n"
  CUDA_VISIBLE_DEVICES="-1"
  nohup ./env/bin/python src/models/main_spn_structure.py \
    --result-dir $result_dir \
    --batch-size 1024 \
    --n-gaussians $n \
    --lr 0.001 \
    --net "spn" \
    --epochs 100 \
    --njobs 3 \
    --n-labels 5 \
    --seed 0 \
    --experiment-name $name &

done
