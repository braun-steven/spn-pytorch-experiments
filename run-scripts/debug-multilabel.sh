set -e


# Hyperparameters
batch_size=128
test_batch_size=128
lr=0.01
epochs=50
njobs=4
n_labels=10
n_digits=10
canvas_size=50
seed=0
resnet_arch="resnet18"
arch="pure-spn"
cuda_device_id="0"

# Create base directory based on the current time
base_dir="./results/debug"
TAG="debug"
result_dir="$base_dir/$TAG"
mkdir -p $result_dir

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
  --cuda-device-id $cuda_device_id \
  --net $arch
