# Image-augmentation
Reproduce Manifold Mixup(Verma, Vikas, et al.) and AugMix(Hendrycks, Dan, et al.)

## TODOs

- [ ] PreActResNet-20 코드에 오류 없는지 검토 필요
- [ ] mixup, augmix 폴더를 utils로 rename하고, main.py를 하나로 통합하기
  - script에서 실험 옵션 세부적으로 적용할 수 있도록 수정

## How to Run

You can run the script file below to select experimental options.

```shell
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
| ResNet-20                                    |           | -                   |                    |
| ResNet-18                                    |           | -                   |                    |
| Manifold Mixup, PreActResNet-20              |           | -                   |                    |
| Manifold Mixup, PreActResNet-18 (2000 epoch) |           | 79.66% (1200 epoch) |                    |
| Manifold Mixup, PreActResNet-18 (200 epoch)  |           | 79.66% (1200 epoch) |                    |
| AugMix, PreActResNet-20                      |           | -                   |                    |
| AugMix, PreActResNet-18                      |           | -                   |                    |

## References

- Manifold Mixup: https://github.com/vikasverma1077/manifold_mixup
- AugMix: https://github.com/google-research/augmix
- AugMix torchvision: https://pytorch.org/vision/main/generated/torchvision.transforms.AugMix.html#torchvision.transforms.AugMix
