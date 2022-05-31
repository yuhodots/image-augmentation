# Image-augmentation
Reproduce Manifold Mixup(Verma, Vikas, et al.) and AugMix(Hendrycks, Dan, et al.)

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

### CIFAR-100

|                                              | Reproduce | Paper               | Clustering quality |
| -------------------------------------------- | --------- | ------------------- | ------------------ |
| ResNet-20 (200 epoch)                        |           | -                   |                    |
| ResNet-18 (200 epoch)                        |           | -                   |                    |
| Manifold Mixup, PreActResNet-20 (2000 epoch) |           | -                   |                    |
| Manifold Mixup, PreActResNet-18 (2000 epoch) |           | 79.66% (1200 epoch) |                    |
| Manifold Mixup, PreActResNet-18 (200 epoch)  |           | 79.66% (1200 epoch) |                    |
| AugMix, PreActResNet-20 (200 epoch)          |           | -                   |                    |
| AugMix, PreActResNet-18 (200 epoch)          |           | -                   |                    |

## References

- Manifold Mixup: https://github.com/vikasverma1077/manifold_mixup
- AugMix: https://github.com/google-research/augmix
- AugMix torchvision: https://pytorch.org/vision/main/generated/torchvision.transforms.AugMix.html#torchvision.transforms.AugMix
