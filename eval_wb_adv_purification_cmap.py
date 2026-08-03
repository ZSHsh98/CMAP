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
import torch
import random
import numpy as np

import utils
from utils import str2bool, load_data, load_adversarial_data, diffusion_to_uniform, diffusion_to_gaussian, initialize_from_gaussian, purify_cifar, purify_imagenet, get_accuracy, get_accuracy_vote
from load_models import get_image_classifier, get_diffusion_model, get_consistency_model
from ensattack import ens_attack

from tqdm import tqdm
import datetime
import pickle
import pytorch_ssim

def adversarial_generate(args, config):
    assert args.domain in ["cifar10", "imagenet100"]

    adversarial_dir = '_'.join(['Adversarial_data_noise', args.classifier_name, 'seed' + str(args.seed), 'data' + str(args.data_seed)])
    os.makedirs(adversarial_dir, exist_ok=True)

    logger = utils.Logger(file_name=f'{adversarial_dir}/{args.attack_method}_{str(args.atk_t_cm)}_{str(args.epsilon)}_{str(args.num_sub)}_generation.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and classifier...')
    
    if args.domain in ['cifar10']:
        cm_model = get_consistency_model(args.domain)
        args.class_num = 10
    elif args.domain in ['imagenet100']:
        cm_model, cm_diffusion = get_consistency_model(args.domain)
        args.class_num = 100

    classifier = get_image_classifier(args.classifier_name).to(config.device)

    # load data
    print('starting the dataloader...')
    args.datapath = './dataset' if args.domain == 'cifar10' else '/home/student.unimelb.edu.au/jiahaoyang/IMIA/add_storage_500/imagenet100'
    loader = load_data(args, adv_batch_size)
    
    if config.data.dataset == 'CIFAR10':
        args.num_classes = 10
        mean = torch.from_numpy(np.array([x / 255
            for x in [125.3, 123.0, 113.9]])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array([x / 255
            for x in [63.0, 62.1, 66.7]])).view(1,3,1,1).cuda().float()
    elif config.data.dataset == 'CIFAR100':
        args.num_classes = 100
        mean = torch.from_numpy(np.array([x / 255
            for x in [129.3, 124.1, 112.4]])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array([x / 255
            for x in [68.2, 65.4, 70.4]])).view(1,3,1,1).cuda().float()
    elif config.data.dataset == 'ImageNet100':
        args.num_classes = 100
        mean = torch.from_numpy(np.array(
            [0.485, 0.456, 0.406])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array(
            [0.229, 0.224, 0.225])).view(1,3,1,1).cuda().float()
    elif config.data.dataset == 'ImageNet':
        args.num_classes = 1000
        mean = torch.from_numpy(np.array(
            [0.485, 0.456, 0.406])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array(
            [0.229, 0.224, 0.225])).view(1,3,1,1).cuda().float()
    
    print('reset the mean 0 and std 1')
    mean = mean - mean
    std = std / std

    # args.num_steps = 5
    args.step_size_adv = args.epsilon / args.num_steps

    print('generating the adversarial data...')    
    data_path = adversarial_dir + '/' + '_'.join([args.attack_method, str(args.atk_t_cm), str(args.epsilon), str(args.num_sub)]) + '.pkl'

    x_list = []
    X_adv_list = []
    y_list = []
    print('---------------- apply no attack to classifier ----------------')
    acc = 0.
    for i, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)

        acc += get_accuracy(classifier, x, y)
    
    total_acc = acc / math.ceil(args.num_sub/args.adv_batch_size)

    print(f'clean_acc: {"%.4f" %total_acc}')
        
    print('------------ apply surrogate attack to classifier -------------')
    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'), f' | Classifier: {args.classifier_name} | Dataset: {args.domain} | Attack: {args.attack_method} | epsilon: {args.epsilon}')
    for i, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)
        acc = get_accuracy(classifier, x, y)

        if args.epsilon == 0:
            X_adv = x
            print(f'batch: {i + 1};  \tclean_acc: {"%.4f" %acc}')
        else:
            if args.domain in ['cifar10']:
                X_adv, top1, top5 = ens_attack(x, y, cm_model, classifier, mean, std, args, args.attack_method, config)
            elif args.domain in ['imagenet100']:
                X_adv, top1, top5 = ens_attack(x, y, cm_model, classifier, mean, std, args, args.attack_method, config, cm_diffusion=cm_diffusion)
            print(f'batch: {i + 1};  \tclean_acc: {"%.4f" %acc};  \trobust_acc(top1): {"%.4f" %top1};  \trobust_acc(top5): {"%.4f" %top5}')
        x_list.append(x)
        X_adv_list.append(X_adv)
        y_list.append(y)

    x = torch.cat(x_list, dim=0)
    X_adv = torch.cat(X_adv_list, dim=0)
    y = torch.cat(y_list, dim=0)
    print(x.shape, X_adv.shape)

    adversarial_data = dict(X_adv = X_adv, y = y)
    with open(data_path, "wb") as f:
        pickle.dump(adversarial_data, f)
    
    logger.close()


def purify_optimize(args, config):
    start_time = time.time()
    print('loading the consistency model...')
    if args.domain in ['cifar10']:
        cm_model = get_consistency_model(args.domain)
        class_num = 10
    elif args.domain in ['imagenet100']:
        cm_model, cm_diffusion = get_consistency_model(args.domain)
        class_num = 100

    cm_model.to(config.device)
    cm_model.eval()

    print('loading the classifier...')
    classifier = get_image_classifier(args.classifier_name).to(config.device)

    adversarial_dir = '_'.join(['Adversarial_data_noise', args.classifier_name, 'seed' + str(args.seed), 'data' + str(args.data_seed)])
    data_path = adversarial_dir + '/' + '_'.join([args.attack_method, str(args.atk_t_cm), str(args.epsilon), str(args.num_sub)]) + '.pkl'
    logger = utils.Logger(file_name=f'{adversarial_dir}/{args.attack_method}_{str(args.atk_t_cm)}_{str(args.epsilon)}_{str(args.num_sub)}_purification.txt', file_mode="w+", should_flush=True)

    print(f'loading the adversarial data from {data_path}')
    with open(data_path, "rb") as f:
        adversarial_data = pickle.load(f)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    adversarial_loader = load_adversarial_data(adversarial_data['X_adv'], adversarial_data['y'], adv_batch_size)
    
    batch_num_list = []
    pur_acc_rdm_list = torch.empty(math.ceil(args.num_sub/args.adv_batch_size), args.iterations).to(config.device)
    pur_acc_soft_list = torch.empty(math.ceil(args.num_sub/args.adv_batch_size), args.iterations).to(config.device)
    pur_acc_hard_list = torch.empty(math.ceil(args.num_sub/args.adv_batch_size), args.iterations).to(config.device)

    acc_num_rdm = 0
    acc_num_soft = 0
    acc_num_hard = 0
    
    Z_mean_all = []
    Z_std_all = []
    x_pur_list = []

    for i, (x, y) in enumerate(adversarial_loader):
        x, y = x.to(config.device), y.to(config.device)
        # repeat adversarial samples k times
        X_adv_k_samples, y = x.repeat(args.k_samples,1,1,1,1), y.repeat(args.k_samples,1)

        batch_num_list.append(x.shape[1])
        X_init = initialize_from_gaussian(X_adv_k_samples, args, config)

        X_init.requires_grad_()
        optimizer = torch.optim.Adam([X_init], lr=args.lr, betas=(0.9,0.99))
        loss_func1 = torch.nn.L1Loss(reduction='sum')
        loss_func2 = torch.nn.MSELoss(reduction='sum')
        
        Z_mean_iteration = []
        Z_std_iteration = []

        # purfy X_diff to X_pur
        for iter in tqdm(range(args.iterations), desc=f'Purification of batch {i+1}(/{math.ceil(args.num_sub/args.adv_batch_size)})', position=0):
            with torch.enable_grad():
                X_diff_k_samples = X_adv_k_samples*2-1

                X_pur_k_samples = torch.empty((X_init.shape)).to(config.device)
                ssim_loss = 0

                K, bs, c, h, w = X_init.shape
                X_init_flat = X_init.reshape(K * bs, c, h, w)

                if args.domain in ['cifar10']:
                    X_pur_flat = purify_cifar(cm_model, X_init_flat, 80, config.device)
                elif args.domain in ['imagenet100']:
                    X_pur_flat = purify_imagenet(X_init_flat, cm_model, cm_diffusion, 80, config.device)

                X_pur_k_samples = X_pur_flat.reshape(K, bs, c, h, w)
                X_diff_flat = X_diff_k_samples.reshape(K * bs, c, h, w)

                ssim_loss = ssim_loss - pytorch_ssim.ssim((X_pur_flat + 1) * 0.5, (X_diff_flat + 1) * 0.5, batch_sum=True)

                gaussian_mean = torch.zeros_like(X_init[0])
                gaussian_std = 80 * torch.ones_like(X_init[0])
                loss = args.scale_factor * (loss_func1(X_pur_k_samples, X_diff_k_samples) + args.similar_factor * ssim_loss + args.gauss_factor * args.k_samples * (loss_func2(X_init.mean(dim=0), gaussian_mean) + loss_func2(X_init.std(dim=0), gaussian_std)))
                # loss = 1/ X_init.shape[1] * (loss_func1(X_pur_k_samples, x*2-1,dim=1).sum() + args.similar_factor * (ssim_loss.sum() / args.k_samples) + args.gauss_factor * (loss_func2(X_init.mean(dim=0), gaussian_mean,dim=0).sum() + loss_func2(X_init.std(dim=0), gaussian_std,dim=0)).sum())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_rdm = []
            for j in range(args.k_samples):
                X_pur_rdm_samples = torch.cat([X_pur_k_samples[random.randint(0, args.k_samples-1)][n].unsqueeze(0) for n in range(adv_batch_size)])
                acc_rdm.append(float(get_accuracy(classifier, (X_pur_rdm_samples + 1) * 0.5, y[0])))
            acc_rdm_avg = np.mean(acc_rdm)
            acc_vote_soft = get_accuracy_vote(classifier, (X_pur_k_samples + 1) * 0.5, y, 'soft', class_num=class_num)
            acc_vote_hard = get_accuracy_vote(classifier, (X_pur_k_samples + 1) * 0.5, y, 'hard', class_num=class_num)

            pur_acc_rdm_list[i][iter] = (float(acc_rdm_avg))
            pur_acc_soft_list[i][iter] = (float(acc_vote_soft))
            pur_acc_hard_list[i][iter] = (float(acc_vote_hard))

            Z_mean = X_init.mean(dim=0).mean()
            Z_std = X_init.std(dim=0).mean()

            Z_mean_iteration.append(float(Z_mean.cpu()))
            Z_std_iteration.append(float(Z_std.cpu()))

        acc_rdm = []
        for j in range(args.k_samples):
            X_pur_rdm_samples = torch.cat([X_pur_k_samples[random.randint(0, args.k_samples-1)][n].unsqueeze(0) for n in range(adv_batch_size)])
            acc_rdm.append(float(get_accuracy(classifier, (X_pur_rdm_samples + 1) * 0.5, y[0])))
        acc_rdm_avg = np.mean(acc_rdm)
        acc_soft = get_accuracy_vote(classifier, (X_pur_k_samples + 1) * 0.5, y, 'soft', class_num=class_num)
        acc_hard = get_accuracy_vote(classifier, (X_pur_k_samples + 1) * 0.5, y, 'hard', class_num=class_num)

        # print(f"max: {X_pur_k_samples.max()}; min: {X_pur_k_samples.min()}")
        print(f"batch: {i + 1}; \tpur_acc(soft):{round(float(acc_soft),4)}; \tpur_acc(hard):{round(float(acc_hard),4)}; \tpur_acc(rdm):{acc_rdm_avg}")
        # print(f"batch: {i + 1}; \tpur_acc(soft):{round(float(acc_soft),4)}; \tpur_acc(hard):{round(float(acc_hard),4)}")

        acc_num_rdm = acc_num_rdm + acc_rdm_avg * X_pur_k_samples.shape[1]
        acc_num_soft = acc_num_soft + acc_soft * X_pur_k_samples.shape[1]
        acc_num_hard = acc_num_hard + acc_hard * X_pur_k_samples.shape[1]

        Z_mean_all.append(Z_mean_iteration)
        Z_std_all.append(Z_std_iteration)

        x_pur_list.append(X_pur_k_samples[0].detach().cpu())
        
    end_time = time.time()

    Z = dict(Z_mean = Z_mean_all, Z_std = Z_std_all)

    # out_dir = f'Plot_Z/CIFAR10_{args.similar_factor}_{args.gauss_factor}'
    # os.makedirs(out_dir, exist_ok=True)
    # with open(os.path.join(out_dir, f'Z_curve.pkl'), 'wb') as f:
    #     pickle.dump(Z, f)

    all_x_pur = torch.cat(x_pur_list, dim=0).cpu()
    print(all_x_pur.shape)
    
    acc_all_rdm = acc_num_rdm / args.num_sub
    acc_all_soft = acc_num_soft / args.num_sub
    acc_all_hard = acc_num_hard / args.num_sub

    print(f'purfied_acc_soft (500): {"%.4f" %acc_all_soft}\npurfied_acc_hard (500): {"%.4f" %acc_all_hard}\npurfied_acc_rdm (500): {acc_all_rdm}\n')
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
    parser.add_argument('--attack_version', type=str, default='standard')
    
    # addition
    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--num_steps', default=100, type=int,help='perturb number of steps')
    parser.add_argument('--attack_method', type=str, default='PGD', choices=['AA_Attack', 'AA_Attack_L2', 'PGD_EOT','PGD_EOT_L2','BPDA_EOT'])
    parser.add_argument('--vote_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--epsilon', type=float, default=0.01569)
    parser.add_argument('--lr', type=float, default=2, help='learning rate')
    parser.add_argument('--k_samples', type=int, default=10)
    parser.add_argument('--scale_factor', type=float, default=0.0001)
    parser.add_argument('--similar_factor', type=float, default=1)
    parser.add_argument('--gauss_factor', type=float, default=0.00002)
    parser.add_argument('--iterations', type=int, default=1000)
    # indensity
    parser.add_argument('--prob', default=0.5, type=float, help='probability of using diverse inputs')
    parser.add_argument('--image_resize', default=331, type=int, help='heigth of each input image')
    # PGD & PGD_L2
    parser.add_argument('--random', default=True,help='random initialization for PGD')

    parser.add_argument('--atk_t_cm', type=int, default=5, help='Sampling noise scale')

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
