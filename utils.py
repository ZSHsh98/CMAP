# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys
import argparse
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import data

import numpy as np


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def display_distance(x):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=3, init='pca', random_state=81)
    # tsne = TSNE(n_components=2, init='random', random_state=0)
    color = ['g','b','r','c','m','y','k','coral','aqua','pink']
    size = [10+0.05*n for n in range(len(x))]
    
    # (iteration, k_samples, batch_size, 3, 32, 32)
    feature = []
    for iter in range(len(x)):
        for k in range(len(x[0])):
            feature.append(x[iter][k][0].cpu().detach().numpy())

    feature = np.array(feature)
    print('shape of features:',feature.shape)
    tsne_feature = tsne.fit_transform(feature.reshape(feature.shape[0], feature.shape[1]*feature.shape[2]*feature.shape[3]))
    print('shape of tsne_features:',tsne_feature.shape)

    
    ax = plt.subplot(projection = '3d')
    ax.grid(False)

    for iter in range(len(x)):
        if iter % 10 == 0:
            for k in range(len(x[0])):
                ax.scatter(tsne_feature[int(iter*len(x[0])+k),0], tsne_feature[int(iter*len(x[0])+k),1], tsne_feature[int(iter*len(x[0])+k),2], c=color[k], s=size[iter])
    
    plt.savefig('./tempresult/cifar10/L1_SSIMLoss/distance_1000_3d.png')

def get_accuracy(model, x_orig, y_orig, device=torch.device('cuda:0')):
    acc = 0.
    x = x_orig.clone().to(device)
    y = y_orig.clone().to(device)
    output = model(x)
    acc = acc + (output.max(1)[1] == y).float().sum()

    return acc / x_orig.shape[0]

def get_accuracy_vote(model, x, y, vote_type='soft', device=torch.device('cuda:0'), class_num=10):
    import torch.nn.functional as F
    acc = 0.
    output = torch.zeros(x.shape[1],class_num).to(device)
    if vote_type == 'soft':
        for i in range(len(x)):
            output += F.softmax(model(x[i]), dim=0)
        acc = acc + (output.max(1)[1] == y[0]).float().sum()
    elif vote_type == 'hard':
        for i in range(len(x)):
            output += F.one_hot(model(x[i]).max(1)[1], class_num)
        acc = acc + (output.max(1)[1] == y[0]).float().sum()

    return acc / x[0].shape[0]


def load_data(args, adv_batch_size):
    if 'imagenet224' in args.domain:
        data_dir = args.datapath
        transform = data.get_transform('imagenet64', 'imval', base_size=64)
        val_data = data.imagent_dataset_sub(data_dir, transform=transform,
											num_sub=args.num_sub, data_seed=args.data_seed)
        loader = DataLoader(val_data, batch_size=adv_batch_size, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))
    elif 'imagenet100' in args.domain:
        data_dir = args.datapath
        transform = data.get_transform(args.domain, 'imval', base_size=64)
        val_data = data.imagent_dataset_sub(data_dir, transform=transform,
											num_sub=args.num_sub, data_seed=args.data_seed)
        loader = DataLoader(val_data, batch_size=adv_batch_size, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))
    elif 'cifar10' in args.domain:
        data_dir = args.datapath
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = data.cifar10_dataset_sub(data_dir, transform=transform,
                                            num_sub=args.num_sub, data_seed=args.data_seed)
        loader = DataLoader(val_data, batch_size=adv_batch_size, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))
    else:
        raise NotImplementedError(f'Unknown domain: {args.domain}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return loader

def load_adversarial_data(x, y, adv_batch_size):
    class GetLoader(torch.utils.data.Dataset):
        def __init__(self, data_root, data_label):
            self.data = data_root
            self.label = data_label
        def __getitem__(self, index):
            data = self.data[index]
            labels = self.label[index]
            return data, labels
        def __len__(self):
            return len(self.data)
    
    adversarial_dataset = GetLoader(x.cpu().numpy(), y.cpu().numpy())
    loader = DataLoader(adversarial_dataset, batch_size=adv_batch_size, shuffle=False, pin_memory=True, num_workers=4)
    x_val, y_val = next(iter(loader))

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return loader

def diffusion_to_uniform(X_adv, radius, config):
    e = torch.rand_like(X_adv)
    print(f'noise max: {(radius * 2 * (e - 0.5)).max()}; min: {(radius * 2 * (e - 0.5)).min()}')
    X_diff = X_adv + radius * 2 * (e - 0.5)

    return X_diff

def diffusion_to_gaussian(X_adv, t, config):
    sigma_max, sigma_min, rho = 80, 0.002, 7
    total_noise_levels = 18
    noise_levels = t

    e = torch.randn_like(X_adv)
    step_indices = torch.arange(total_noise_levels, dtype=torch.float64, device=config.device)
    # sigma_steps = (sigma_max ** (1 / rho) + (total_noise_levels - 1 - step_indices) / (total_noise_levels - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_steps = (sigma_min**(1/rho) + step_indices / (total_noise_levels - 1) * (sigma_max**(1/rho) - sigma_min**(1/rho)))**rho
    X_diff = X_adv + sigma_steps[noise_levels - 1] * e

    return sigma_steps[noise_levels - 1], X_diff

def initialize_from_gaussian(X_adv, args, config):
    sigma_max, sigma_min, rho = 80, 0.002, 7
    total_noise_levels = 1000
    noise_levels = args.t

    e = torch.randn_like(X_adv)
    step_indices = torch.arange(total_noise_levels, dtype=torch.float64, device=config.device)
    # sigma_steps = (sigma_max ** (1 / rho) + (total_noise_levels - 1 - step_indices) / (total_noise_levels - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # X_init = (sigma_steps[noise_levels - 1]**2 - sigma_min**2).sqrt() * e
    sigma_steps = (sigma_min**(1/rho) + step_indices / (total_noise_levels - 1) * (sigma_max**(1/rho) - sigma_min**(1/rho)))**rho
    X_init = sigma_steps[noise_levels - 1] * e

    return X_init


def purify_cifar(model, images, noise, device, use_ema: bool = False):
    from cm_cifar10.jcm import sde_lib
    from cm_cifar10.jcm.models import utils as mutils
    from cm_cifar10.configs.cifar10_ve_cd import get_config

    config_cm = get_config()
    sde = sde_lib.get_sde(config_cm)

    def get_onestep_sampler(model, init_std=config_cm.sampling.std):
            def sampler(z):
                x = z
                model_fn = mutils.get_distiller_fn(
                    sde,
                    model,
                    train=False,
                    return_state=False,
                )
                std_tensor = torch.ones((x.shape[0],), device=x.device) * init_std
                samples = model_fn(x, std_tensor)
                return samples, 1
            return sampler
        
    onestep_sampler = get_onestep_sampler(model, init_std=noise)
    images = onestep_sampler(images)[0].clip(-1,1)
    
    return images

def purify_imagenet(x_T, model, diffusion, noise, device, sigma_min=0.002, sigma_max=80, rho=7.0, clip_denoised=True, T=1000, model_kwargs={}):
    s_in = x_T.new_ones([x_T.shape[0]])
    # print(f"{sigmas[T-0]}; {sigmas[T-timestep]}; {sigmas[T-T]}")
    _, x_0 = diffusion.denoise(model, x_T, noise * s_in, **model_kwargs)
    return x_0.clamp(-1, 1)
