import torchsummary
from torchvision.models import resnet18 as resnet18_torchvision
from resnet import resnet18 as resnet18_reproduce
from resnet20 import resnet20
from preresnet import preactresnet18
from preresnet20 import preactresnet20


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # ResNet-18
    torchsummary.summary(resnet18_torchvision(num_classes=100), (3, 32, 32), device='cpu')
    torchsummary.summary(resnet18_reproduce(num_classes=100), (3, 32, 32), device='cpu')
    torchsummary.summary(preactresnet18(num_classes=100), (3, 32, 32), device='cpu')

    # ResNet-20
    torchsummary.summary(resnet20(num_classes=100), (3, 32, 32), device='cpu')
    torchsummary.summary(preactresnet20(num_classes=100), (3, 32, 32), device='cpu')


if __name__ == "__main__":
    main()
