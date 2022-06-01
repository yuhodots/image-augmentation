# Image-augmentation
- Reproduce Manifold Mixup(Verma, Vikas, et al.) and AugMix(Hendrycks, Dan, et al.)
- Only CIFAR-100 dataset is supported now.

## How to Run

1. CIFAR-100 with Manifold Mixup

```shell
cd mixup
./run.sh
```

2. CIFAR-100 with AugMix

```shell
cd augmix
./run.sh
```

You can check the summary of model architecture by using `models/summary.py` file.

```shell
cd models
python summary.py
```

## Results

![img](assets/cifar-result.png)

| Experiment option                                                   | Reproduce | Paper               | Clustering quality |
|---------------------------------------------------------------------|-----------|---------------------| ------------------ |
| ResNet-20<br/>(240 epoch)                                           | 69.95%    | -                   |                    |
| ResNet-18<br/>(240 epoch)                                           | TBU       | -                   |                    |
| Manifold Mixup, PreActResNet-20<br>(2000 epoch, original_repo_ver)  | TBU       | -                   |                    |
| Manifold Mixup, PreActResNet-18<br/>(2000 epoch, original_repo_ver) | 80.72%    | 79.66% (1200 epoch) |                    |
| Manifold Mixup, PreActResNet-18<br/>(200 epoch, fast_ver)           | 78.06%    | 79.66% (1200 epoch) |                    |
| AugMix, PreActResNet-20<br/>(200 epoch)                             | TBU       | -                   |                    |
| AugMix, PreActResNet-18<br/>(200 epoch)                             | 75.07%    | -                   |                    |

## References

- Manifold Mixup code is based on https://github.com/vikasverma1077/manifold_mixup
- AugMix code is based on  https://github.com/google-research/augmix
