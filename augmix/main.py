# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import sys
import inspect
import shutil
import time

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from augmix.augmentations import augmentations, augmentations_all
import models
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar100',
                    choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--data_dir', type=str, default='cifar10', help='dataset directory path')
parser.add_argument('--arch', metavar='ARCH', default='preactresnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: preactresnet18)')
parser.add_argument('--memo', type=str, default='')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval_batch_size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

# WRN Architecture options
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen_factor', default=2, type=int, help='Widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')

# AugMix options
parser.add_argument('--mixture_width', default=3, type=int,
                    help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture_depth', default=-1, type=int,
                    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug_severity', default=3, type=int, help='Severity of base augmentation operators')
parser.add_argument('--no_jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
parser.add_argument('--all_ops', '-all', action='store_true',
                    help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Checkpointing options
parser.add_argument('--result_dir', '-s', type=str, default='results/augmix/', help='Folder to save checkpoints.')
parser.add_argument('--print_freq', type=int, default=50, help='Training loss print frequency (batches).')

# Acceleration
parser.add_argument('--num_workers', type=int, default=4, help='Number of pre-fetching threads.')

args = parser.parse_args()


def experiment_name_non_mnist(dataset='cifar100',
                              arch='',
                              epochs=400,
                              batch_size=64,
                              lr=0.01,
                              momentum=0.5,
                              decay=0.0005,
                              add_name=''):
    exp_name = dataset
    exp_name += '_arch_' + str(arch)
    exp_name += '_eph_' + str(epochs)
    exp_name += '_bs_' + str(batch_size)
    exp_name += '_lr_' + str(lr)
    exp_name += '_mom_' + str(momentum)
    exp_name += '_decay_' + str(decay)
    if add_name != '':
        exp_name += '_add_name_' + str(add_name)
    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations
    if args.all_ops:
        aug_list = augmentations_all

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess), aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def train(net, train_loader, optimizer):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.no_jsd:
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
        else:
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), \
                                      F.softmax(logits_aug1, dim=1), \
                                      F.softmax(logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss.backward()
        optimizer.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        if i % args.print_freq == 0:
            print('Train Loss {:.3f}'.format(loss_ema))

    return loss_ema


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(
        test_loader.dataset)


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    # Load datasets
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 100

    train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # Create model
    net = models.__dict__[args.arch](num_classes).cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    start_epoch = 0

    # Result path
    exp_name = experiment_name_non_mnist(dataset=args.dataset, arch=args.arch, epochs=args.epochs,
                                         batch_size=args.batch_size, lr=args.learning_rate, momentum=args.momentum,
                                         decay=args.decay, add_name=args.memo)
    exp_dir = args.result_dir + exp_name
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.isdir(exp_dir):
        raise Exception('%s is not a dir' % exp_dir)

    log_path = os.path.join(exp_dir, args.dataset + '_' + args.arch + '_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    best_acc = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        begin_time = time.time()
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        train_loss_ema = train(net, train_loader, optimizer)
        test_loss, test_acc = test(net, test_loader)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': args.arch,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(exp_dir, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(exp_dir, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                train_loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f} | LR {5}'
              .format((epoch + 1), int(time.time() - begin_time), train_loss_ema, test_loss,
                      100 - 100. * test_acc, current_learning_rate))


if __name__ == '__main__':
    main()
