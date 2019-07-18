# Results

## Batch Sizes
- `lr=1e-4`
- `l2=1e-3`

### bs=64
![](./190704_1604_gridsearch/bs=64_lr=0.0001_l2=0.001/plots/064.png)
### bs=128
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0.001/plots/128.png)
### bs=256
![](./190704_1604_gridsearch/bs=256_lr=0.0001_l2=0.001/plots/256.png)

## Learning Rate
- `bs=128`
- `l2=1e-3`

### lr=1e-3
![](./190704_1604_gridsearch/bs=128_lr=0.001_l2=0.001/plots/128.png)

### lr=1e-4
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0.001/plots/128.png)

## L2 Regularizer
- `bs=128`
- `lr=1e-4`

### l2=1e-1
- Regularization too high
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0.1/plots/128.png)

### l2=1e-2
- Regularization too high
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0.01/plots/128.png)

### l2=1e-3
- Regularization improves Resnet but worsens Resnet+SPN (comp `l2=0` below)
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0.001/plots/128.png)

### l2=1e-4
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0.0001/plots/128.png)

### l2=0
- No regularization works best for Resnet+SPN but worse for Resnet
![](./190704_1604_gridsearch/bs=128_lr=0.0001_l2=0/plots/128.png)
