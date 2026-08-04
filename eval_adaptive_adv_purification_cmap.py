# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import logging
import yaml
import os
import time
import math

import random
import numpy as np

import torch
import torch.nn.functional as F

import utils
from utils import str2bool, load_data, load_adversarial_data, diffusion_to_uniform, diffusion_to_gaussian, initialize_from_gaussian, purify_cifar, purify_imagenet, get_accuracy, get_accuracy_vote
from load_models import get_image_classifier, get_diffusion_model, get_consistency_model

from tqdm import tqdm
import datetime
import pickle
import pytorch_ssim

def adversarial_generate(args, config):  
    print('loading the consistency model...')    
    if args.domain in ['cifar10']:
        cm_model = get_consistency_model(args.domain)
        args.class_num = 10
        args.pur_iteration = 200
    elif args.domain in ['imagenet100']:
        cm_model, cm_diffusion = get_consistency_model(args.domain)
        args.class_num = 100
        args.pur_iteration = 300

    cm_model.to(config.device)
    cm_model.eval()
    
    print('loading the classifier...')
    classifier = get_image_classifier(args.classifier_name).to(config.device)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load data
    print('loading data...')
    args.datapath = './dataset' if args.domain == 'cifar10' else '../imagenet100'
    loader = load_data(args, adv_batch_size)

    adversarial_dir = '_'.join(['Adaptive_Attack_data_noise', args.classifier_name, 'seed' + str(args.seed), 'data' + str(args.data_seed)])
    os.makedirs(adversarial_dir, exist_ok=True)

    print('generating the adaptive adversarial data...')
    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'), f' | Classifier: {args.classifier_name} | Dataset: {args.domain} | Norm: {args.lp_norm} | epsilon: {args.epsilon}')
    data_path = adversarial_dir + '/' + '_'.join([args.lp_norm, str(args.epsilon), str(args.adv_factor), str(args.num_sub)]) + '.pkl'

    logger = utils.Logger(file_name=f'{adversarial_dir}/{args.lp_norm}_{str(args.epsilon)}_{str(args.adv_factor)}_{str(args.num_sub)}_generation.txt', file_mode="w+", should_flush=True)

    X_adv_list = []
    y_list = []
    
    batch_num_list = []
    clean_acc_list = []
    adv_acc_list = []

    for i, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)
        target = F.one_hot(y, args.class_num)
        # repeat adversarial samples k times
        # x (0, 1)
        x, y, target = x.repeat(args.k_samples,1,1,1,1), y.repeat(args.k_samples,1), target.repeat(args.k_samples,1,1)

        batch_num_list.append(x.shape[1])

        # sample from gaussian
        Z = initialize_from_gaussian(x * 2 - 1, args, config)
        sigma_max, sigma_min, rho = 80, 0.002, 7

        Z.requires_grad_()
        optimizer = torch.optim.Adam([Z], lr=args.lr, betas=(0.9,0.99))
        loss_func1 = torch.nn.L1Loss(reduction='sum')
        loss_func2 = torch.nn.MSELoss(reduction='sum')

        # attack X_diff to X_adv
        for iter in range(args.iterations):
            with torch.enable_grad():
                X_adv_k_samples = torch.empty((Z.shape)).to(config.device)
                outputs = torch.empty((target.shape)).to(config.device)
                ssim_loss = 0

                K, bs, c, h, w = Z.shape
                Z_flat = Z.reshape(K * bs, c, h, w)

                if args.domain in ['cifar10']:
                    X_adv_k_samples_flat = purify_cifar(cm_model, Z_flat, sigma_max, config.device)
                elif args.domain in ['imagenet100']:
                    X_adv_k_samples_flat = purify_imagenet(Z_flat, cm_model, cm_diffusion, sigma_max, config.device)

                X_adv_k_samples = X_adv_k_samples_flat.reshape(K, bs, c, h, w)
                x_flat = x.reshape(K * bs, c, h, w)
                ssim_loss = ssim_loss - pytorch_ssim.ssim((X_adv_k_samples_flat + 1) * 0.5, x_flat, batch_sum=True)

                gaussian_mean = torch.zeros_like(Z[0])
                gaussian_std = sigma_max * torch.ones_like(Z[0])
                if (iter + 1) <= args.pur_iteration:
                    loss = args.scale_factor * (loss_func1(X_adv_k_samples, x*2-1) + args.similar_factor * ssim_loss + args.gauss_factor * args.k_samples * (loss_func2(Z.mean(dim=0), gaussian_mean) + loss_func2(Z.std(dim=0), gaussian_std)))
                else:
                    if args.lp_norm == 'Linf':
                        # (-1, 1) --> (0, 1)
                        X_adv_k_samples = torch.clamp((X_adv_k_samples + 1) * 0.5 - x, -args.epsilon, args.epsilon) + x
                        if (iter + 1) % 200 == 0:
                            print(f'eta max: {(X_adv_k_samples - x).max()}; eta min: {(X_adv_k_samples - x).min()}')
                    elif args.lp_norm == 'L2':
                        eta = args.epsilon * ((X_adv_k_samples + 1) * 0.5 - x) / torch.norm(((X_adv_k_samples + 1) * 0.5 - x).view(args.k_samples, adv_batch_size, -1), dim=2).view(args.k_samples, adv_batch_size, 1, 1, 1)
                        if (iter + 1) % 200 == 0:
                            print(f'eta: {(torch.norm(eta.view(args.k_samples, adv_batch_size, -1), dim=2).view(args.k_samples, adv_batch_size, 1, 1, 1)).max()}')
                        X_adv_k_samples = x + eta
                    # (0, 1)
                    X_adv_k_samples.clamp(0, 1.0)
                    for k in range(args.k_samples):
                        outputs[k] = F.softmax(classifier(X_adv_k_samples[k]))
                    # L1 shoule be (-1, 1)
                    # loss = - args.adv_factor * args.k_samples * F.cross_entropy(outputs, target.float(), reduction='sum') +  args.scale_factor * (loss_func1(X_adv_k_samples, x) + args.similar_factor * ssim_loss + args.gauss_factor * args.k_samples * (loss_func2(Z.mean(dim=0), gaussian_mean) + loss_func2(Z.std(dim=0), gaussian_std)))
                    loss = - args.adv_factor * args.k_samples * F.cross_entropy(outputs, target.float(), reduction='sum') +  args.scale_factor * (loss_func1(X_adv_k_samples*2-1, x*2-1) + args.similar_factor * ssim_loss + args.gauss_factor * args.k_samples * (loss_func2(Z.mean(dim=0), gaussian_mean) + loss_func2(Z.std(dim=0), gaussian_std)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (iter + 1) <= args.pur_iteration:
                # (-1, 1)
                acc_vote_soft = get_accuracy_vote(classifier, (X_adv_k_samples + 1) * 0.5, y, 'soft', class_num=args.class_num)
                acc_vote_hard = get_accuracy_vote(classifier, (X_adv_k_samples + 1) * 0.5, y, 'hard', class_num=args.class_num)
                if (iter + 1) % 200 == 0:
                    print(f"batch: {i + 1}; \titer:{iter + 1}; \tacc_vote_soft:{round(float(acc_vote_soft),4)}; \tacc_vote_hard:{round(float(acc_vote_hard),4)}")
                if (iter + 1) == args.pur_iteration:
                    clean_acc = max(acc_vote_soft, acc_vote_hard)
                    clean_acc_list.append(float(clean_acc))
            else:
                # (0, 1)
                acc_vote_soft = get_accuracy_vote(classifier, X_adv_k_samples, y, 'soft', class_num=args.class_num)
                acc_vote_hard = get_accuracy_vote(classifier, X_adv_k_samples, y, 'hard', class_num=args.class_num)
                if (iter + 1) % 200 == 0:
                    print(f"batch: {i + 1}; \titer:{iter + 1}; \tclean_acc:{round(float(clean_acc),4)}; \tacc_vote_soft:{round(float(acc_vote_soft),4)}; \tacc_vote_hard:{round(float(acc_vote_hard),4)}")
                if (iter + 1) == args.iterations:
                    adv_acc = min(acc_vote_soft, acc_vote_hard)
                    adv_acc_list.append(float(adv_acc))
        
        # (0, 1)
        X_adv_list.append(X_adv_k_samples.permute(1,0,2,3,4))
        y_list.append(y.permute(1,0))


    X_adv = torch.cat(X_adv_list, dim=0)
    y = torch.cat(y_list, dim=0)

    adversarial_data = dict(X_adv = X_adv, y = y)
    print(X_adv.shape, y.shape)
    print(type(adversarial_data), adversarial_data['X_adv'].shape, adversarial_data['y'].shape)
    with open(data_path, "wb") as f:
        pickle.dump(adversarial_data, f)
    
    clean_acc_avg = 0.
    adv_acc_avg = 0.
    for n in range(len(batch_num_list)):
        clean_acc_avg += batch_num_list[n] * clean_acc_list[n]
        adv_acc_avg += batch_num_list[n] * adv_acc_list[n]
    print(f'batch_num: {batch_num_list}')
    print(f'clean_acc: {clean_acc_list}; \tclean_acc_avg: {clean_acc_avg / args.num_sub}')
    print(f'adv_acc: {adv_acc_list}; \tadv_acc_avg: {adv_acc_avg / args.num_sub}')

    logger.close()


def purify_optimize(args, config):    
    print('loading the consistency model...')    
    if args.domain in ['cifar10']:
        model = get_consistency_model(args.domain)
        class_num = 10
        pur_iteration = 200
    elif args.domain in ['imagenet100']:
        model, diffusion = get_consistency_model(args.domain)
        class_num = 100
        pur_iteration = 400

    # if torch.cuda.device_count() > 1:
    # 	model = nn.DataParallel(model)
    model.to(config.device)
    model.eval()
    
    print('loading the classifier...')
    classifier = get_image_classifier(args.classifier_name).to(config.device)

    ngpus = torch.cuda.device_count()
    args.adv_batch_size = args.adv_batch_size * 2
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load data
    print('loading the adversarial data...')
    adversarial_dir = '_'.join(['AntiODEPure_Attack_data_noise', args.classifier_name, 'seed' + str(args.seed), 'data' + str(args.data_seed)])
    data_path = adversarial_dir + '/' + '_'.join([args.lp_norm, str(args.epsilon), str(args.adv_factor), str(args.num_sub)]) + '.pkl'

    logger = utils.Logger(file_name=f'{adversarial_dir}/{args.lp_norm}_{str(args.epsilon)}_{str(args.adv_factor)}_{str(args.num_sub)}_purification.txt', file_mode="w+", should_flush=True)

    with open(data_path, "rb") as f:
        adversarial_data = pickle.load(f)
    
    adversarial_loader = load_adversarial_data(adversarial_data['X_adv'].detach(), adversarial_data['y'].detach(), adv_batch_size)

    batch_num_list = []
    puradv_acc_soft_list = []
    puradv_acc_hard_list = []
    pur_acc_soft_list = []
    pur_acc_hard_list = []
    
    start_time = time.time()

    for i, (x, y) in enumerate(adversarial_loader):
        # X_adv_k_samples, y = x.to(config.device), y.to(config.device)
        X_adv_k_samples, y = x.permute(1,0,2,3,4).to(config.device), y.permute(1,0).to(config.device)
        
        batch_num_list.append(X_adv_k_samples.shape[1])

        # CMAP
        sigma_max, sigma_min, rho = 80, 0.002, 7
        # X_init = initialize_from_gaussian(x * 2 - 1, args, config)
        X_init = initialize_from_gaussian(X_adv_k_samples, args, config)
        X_init.requires_grad_()

        # (-1, 1)
        X_adv_k_samples = X_adv_k_samples.detach().clone()
        optimizer = torch.optim.Adam([X_init], lr=args.lr, betas=(0.9,0.99))
        loss_func1 = torch.nn.L1Loss(reduction='sum')
        loss_func2 = torch.nn.MSELoss(reduction='sum')
        for iter in tqdm(range(pur_iteration), desc=f'Purification of batch {i+1}(/{math.ceil(args.num_sub/args.adv_batch_size)})', position=0):
            # generate the X_pur
            X_adv_k_samples = X_adv_k_samples.detach().clone()
            with torch.enable_grad():
                X_pur_k_samples = torch.empty((X_init.shape)).to(config.device)
                # outputs = torch.empty((target.shape)).to(config.device)
                ssim_loss = 0
                K, bs, c, h, w = X_init.shape
                X_init_flat = X_init.reshape(K * bs, c, h, w)

                if args.domain in ['cifar10']:
                    X_pur_flat = purify_cifar(model, X_init_flat, 80, config.device)
                elif args.domain in ['imagenet100']:
                    X_pur_flat = purify_imagenet(X_init_flat, model, diffusion, 80, config.device)

                X_pur_k_samples = X_pur_flat.reshape(K, bs, c, h, w)
                X_diff_flat = X_adv_k_samples.reshape(K * bs, c, h, w)

                ssim_loss = ssim_loss - pytorch_ssim.ssim((X_pur_flat + 1) * 0.5, (X_diff_flat + 1) * 0.5, batch_sum=True)

                gaussian_mean = torch.zeros_like(X_init[0])
                gaussian_std = sigma_max * torch.ones_like(X_init[0])
                loss = args.scale_factor * (loss_func1(X_pur_k_samples, X_adv_k_samples) + args.similar_factor * ssim_loss + args.gauss_factor * args.k_samples * (loss_func2(X_init.mean(dim=0), gaussian_mean) + loss_func2(X_init.std(dim=0), gaussian_std)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        acc_vote_soft = get_accuracy_vote(classifier, (X_pur_k_samples + 1) * 0.5, y, 'soft', class_num=args.class_num)
        acc_vote_hard = get_accuracy_vote(classifier, (X_pur_k_samples + 1) * 0.5, y, 'hard', class_num=args.class_num)
        pur_acc_soft_list.append(float(acc_vote_soft))
        pur_acc_hard_list.append(float(acc_vote_hard))
        print(f"batch: {i + 1}; \tpur_acc_vote_soft:{round(float(acc_vote_soft),4)}; \tpur_acc_vote_hard:{round(float(acc_vote_hard),4)}")
        
    end_time = time.time()

    puradv_acc_soft_avg = 0.
    puradv_acc_hard_avg = 0.
    pur_acc_soft_avg = 0.
    pur_acc_hard_avg = 0.
    for n in range(len(batch_num_list)):
        puradv_acc_soft_avg += batch_num_list[n] * puradv_acc_soft_list[n]
        puradv_acc_hard_avg += batch_num_list[n] * puradv_acc_hard_list[n]
        pur_acc_soft_avg += batch_num_list[n] * pur_acc_soft_list[n]
        pur_acc_hard_avg += batch_num_list[n] * pur_acc_hard_list[n]
    print(f'batch_num: {batch_num_list}')
    print(f'puradv_acc_soft: {puradv_acc_soft_list}; \tpuradv_acc_soft_avg: {puradv_acc_soft_avg / args.num_sub}')
    print(f'puradv_acc_hard: {puradv_acc_hard_list}; \tpuradv_acc_hard_avg: {puradv_acc_hard_avg / args.num_sub}')
    print(f'pur_acc_soft: {pur_acc_soft_list}; \tpur_acc_soft_avg: {pur_acc_soft_avg / args.num_sub}')
    print(f'pur_acc_hard: {pur_acc_hard_list}; \tpur_acc_hard_avg: {pur_acc_hard_avg / args.num_sub}')

    print(f'time used: {end_time - start_time}')

    logger.close()

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')

    # adv
    parser.add_argument('--domain', type=str, default='cifar10', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='cifar10-wideresnet-28-10', help='which classifier to use')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    
    # addition
    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--num_steps', default=100, type=int,help='perturb number of steps')
    parser.add_argument('--vote_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--epsilon', type=float, default=0.01569)
    parser.add_argument('--lr', type=float, default=2, help='learning rate')
    parser.add_argument('--k_samples', type=int, default=10)
    parser.add_argument('--adv_factor', type=float, default=200)
    parser.add_argument('--scale_factor', type=float, default=0.0001)
    parser.add_argument('--similar_factor', type=float, default=1)
    parser.add_argument('--gauss_factor', type=float, default=0.00002)
    parser.add_argument('--iterations', type=int, default=1000)
    # indensity
    parser.add_argument('--prob', default=0.5, type=float, help='probability of using diverse inputs')
    parser.add_argument('--image_resize', default=331, type=int, help='heigth of each input image')
    # PGD & PGD_L2
    parser.add_argument('--random', default=True,help='random initialization for PGD')


    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    adversarial_generate(args, config)
    purify_optimize(args, config)