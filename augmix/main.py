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

import models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from augmix.utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_command_line_parser():
    parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--data_dir', type=str, default='cifar10', help='dataset directory path')
    parser.add_argument('--arch', metavar='ARCH', default='preactresnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: preactresnet18)')
    parser.add_argument('--partial_class', type=str2bool, default=False, help='use only partial class of dataset')
    parser.add_argument('--partial_class_indices', type=int, default=60)
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
    parser.add_argument('--consistency_loss', type=str, default='none', choices=['none', 'jsd', 'squared_l2'])
    parser.add_argument('--consistency_loss_factor', type=float, default=12)
    parser.add_argument('--all_ops', '-all', action='store_true',
                        help='Turn on all operations (+brightness,contrast,color,sharpness).')

    # Checkpointing options
    parser.add_argument('--result_dir', '-s', type=str, default='results/augmix/', help='Folder to save checkpoints.')
    parser.add_argument('--print_freq', type=int, default=50, help='Training loss print frequency (batches).')

    # Acceleration
    parser.add_argument('--num_workers', type=int, default=4, help='Number of pre-fetching threads.')

    return parser.parse_args()


def normalized_squared_l2_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def load_data(args):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 100

    if args.partial_class:
        num_classes = args.partial_class_indices
        train_data = select_from_default(train_data, np.arange(args.partial_class_indices))
        test_data = select_from_default(test_data, np.arange(args.partial_class_indices))

    train_data = AugMixDataset(args, train_data, preprocess, args.consistency_loss)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def train(args, net, train_loader, optimizer):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.consistency_loss == "none":
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
        elif args.consistency_loss == "jsd":
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), \
                                      F.softmax(logits_aug1, dim=1), \
                                      F.softmax(logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += args.consistency_loss_factor * \
                    (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                     F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                     F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        elif args.consistency_loss == "squared_l2":
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            embeds_all = net(images_all, encode=True)
            embeds_clean, embeds_aug1, embeds_aug2 = torch.split(embeds_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(net.module.linear(embeds_clean), targets)

            embeds_mean = (embeds_clean + embeds_aug1 + embeds_aug2) / 3.
            loss += args.consistency_loss_factor * \
                    (normalized_squared_l2_loss(embeds_mean, embeds_clean).mean() +
                     normalized_squared_l2_loss(embeds_mean, embeds_aug1).mean() +
                     normalized_squared_l2_loss(embeds_mean, embeds_aug2).mean()) / 3.
        else:
            raise AssertionError(f"There is no consistency loss: {args.consistency_loss}")

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

    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def main():
    # Initialization
    torch.manual_seed(1)
    np.random.seed(1)
    args = get_command_line_parser()

    # Load dataset and model
    train_loader, test_loader, num_classes = load_data(args)
    net = models.__dict__[args.arch](num_classes).cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    start_epoch = 0

    # Result path
    exp_name = experiment_name(dataset=args.dataset, arch=args.arch, epochs=args.epochs,
                               batch_size=args.batch_size, lr=args.learning_rate, momentum=args.momentum,
                               decay=args.decay, cl_type=args.consistency_loss, add_name=args.memo)
    exp_dir = args.result_dir + exp_name
    if args.partial_class:
        exp_dir += f"_partial_class_{str(args.partial_class_indices)}"
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

        # Train & Evaluation
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, args.gammas, args.schedule)
        train_loss_ema = train(args, net, train_loader, optimizer)
        test_loss, test_acc = test(net, test_loader)

        # Save checkpoint and best model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {'epoch': epoch, 'dataset': args.dataset, 'model': args.arch, 'state_dict': net.state_dict(),
                      'best_acc': best_acc, 'optimizer': optimizer.state_dict()}
        save_path = os.path.join(exp_dir, 'checkpoint.pth')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(exp_dir, 'model_best.pth'))

        # Save log
        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % ((epoch + 1), time.time() - begin_time,
                                                       train_loss_ema, test_loss, 100 - 100. * test_acc,))

        # Print results
        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f} | LR {5}'
              .format((epoch + 1), int(time.time() - begin_time), train_loss_ema, test_loss,
                      100 - 100. * test_acc, current_learning_rate))


if __name__ == '__main__':
    main()
