import torch
import numpy as np
import random
from augmix.transforms import augmentations, augmentations_all


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise AssertionError("Boolean value is expected")


def experiment_name_non_mnist(dataset='cifar100', arch='', epochs=400, batch_size=64,
                              lr=0.01, momentum=0.5, decay=0.0005, add_name=''):
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


def adjust_learning_rate(args, optimizer, epoch, gammas, schedule):
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


def select_from_default(dataset, index):
    data = dataset.data
    targets = np.asarray(dataset.targets)

    data_tmp = []
    targets_tmp = []
    for i in index:
        ind_cl = np.where(i == targets)[0]
        if data_tmp == []:
            data_tmp = data[ind_cl]
            targets_tmp = targets[ind_cl]
        else:
            data_tmp = np.vstack((data_tmp, data[ind_cl]))
            targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

    dataset.data = data_tmp
    dataset.targets = targets_tmp
    return dataset


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, args, dataset, preprocess, consistency_loss):
        self.args = args
        self.dataset = dataset
        self.preprocess = preprocess
        self.consistency_loss = consistency_loss

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.consistency_loss == "none":
            return self.aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), self.aug(x, self.preprocess), self.aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

    def aug(self, image, preprocess):
        """Perform AugMix augmentations and compute mixture.

        Args:
          image: PIL.Image input image
          preprocess: Preprocessing function which should return a torch tensor.

        Returns:
          mixed: Augmented and mixed image.
        """
        aug_list = augmentations
        if self.args.all_ops:
            aug_list = augmentations_all

        ws = np.float32(np.random.dirichlet([1] * self.args.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(self.args.mixture_width):
            image_aug = image.copy()
            depth = self.args.mixture_depth if self.args.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, self.args.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed
