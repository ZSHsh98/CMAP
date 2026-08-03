import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def gaussian_kernel():
    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st

        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    stack_kernel = stack_kernel.transpose((2, 3, 0, 1))
    stack_kernel = torch.from_numpy(stack_kernel)
    return stack_kernel
    
def smooth(x, stack_kernel):
    ''' implemenet depthwiseConv with padding_mode='SAME' in pytorch '''
    padding = (stack_kernel.size(-1) - 1) // 2
    groups = x.size(1)
    return torch.nn.functional.conv2d(x, weight=stack_kernel, padding=padding, groups=groups)

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().contiguous()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mm_loss(output, target, target_choose, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def ens_attack(input, target, model, classifier , mean, std, args, attack_method, config, attack_classifier=None, cm_diffusion=None):
    def _grad(X, y, mean, std):
        with torch.enable_grad():
                    X.requires_grad_()
                    outputs = attack_classifier(X.sub(mean).div(std))
                    # outputs, __ = model(X.sub(mean).div(std))[0:2]
                    outputs = outputs.softmax(-1)
                    if outputs.dim() == 3:
                        output = outputs.mean(-2) + 1e-10
                    else:
                        output = outputs
                    loss = F.cross_entropy(output.log(), y, reduction='none')
                    grad_ = torch.autograd.grad(
                        [loss], [X], grad_outputs=torch.ones_like(loss),
                        retain_graph=False)[0].detach()
        return grad_

    def input_diversity(args, input_tensor):
        '''apply input transformation to enhance transferability: padding and resizing (DIM)'''
        image_size = 224 if args.domain == 'imagenet' else 32
        rnd = torch.randint(image_size, args.image_resize, ())   # uniform distribution
        rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
        h_rem = args.image_resize - rnd
        w_rem = args.image_resize - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
        if torch.rand(1) < args.prob:
            ret = padded
        else:
            ret = input_tensor
        ret = F.interpolate(ret, [image_size, image_size], mode='nearest')
        return ret
    
    def _PGD_EOT_whitebox(model, cm_diffusion, classifier, X, y, mean, std, eot_step=20):
        from utils import diffusion_to_gaussian, purify_cifar, purify_imagenet
        x_adv = X.detach().clone().to(config.device)
        model.to(config.device)
        for _ in range(args.num_steps):
            grad = torch.zeros_like(x_adv)
            
            for _ in range(eot_step):
                x_adv.requires_grad = True

                with torch.enable_grad():
                    noise, x_diff = diffusion_to_gaussian(x_adv*2-1, args.atk_t_cm, config)

                    if args.domain in ['cifar10']:
                        # (-sigma, sigma) --> (-1, 1)
                        x_pur = purify_cifar(model, x_diff, noise, config.device)
                    elif args.domain in ['imagenet100']:
                        x_pur = purify_imagenet(x_diff, model, cm_diffusion, noise, config.device)
                    
                    # Classification
                    logits = classifier((x_pur+1)*0.5)
                    
                    # Calculate loss
                    loss = F.cross_entropy(logits, y, reduction="sum")
                    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                x_adv = x_adv.detach()

            grad /= eot_step
            grad = grad.sign()
            # x_adv = x_adv + args.num_steps * grad
            x_adv = x_adv + args.step_size_adv * grad

            # Projection
            x_adv = X + torch.clamp(x_adv - X, min=-args.epsilon, max=args.epsilon)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, 0, 1.0)

        print(f'eta max: {(x_adv - X).max()}; eta min: {(x_adv - X).min()}')
        return x_adv
    
    def _PGD_EOT_L2_whitebox(model, cm_diffusion, classifier, X, y, mean, std, eot_step=20):
        from utils import diffusion_to_gaussian, purify_cifar, purify_imagenet
        x_adv = X.detach().clone().to(config.device)
        model.to(config.device)
        
        for _ in range(args.num_steps):
            grad = torch.zeros_like(x_adv)
            
            for _ in range(eot_step):
                x_adv.requires_grad = True
                
                # Classification
                with torch.enable_grad():
                    # Purification module
                    noise, x_diff = diffusion_to_gaussian(x_adv*2-1, args.atk_t_cm, config)

                    if args.domain in ['cifar10']:
                        # (-sigma, sigma) --> (-1, 1)
                        x_pur = purify_cifar(model, x_diff, noise, config.device)
                    elif args.domain in ['imagenet100']:
                        x_pur = purify_imagenet(x_diff, model, cm_diffusion, noise, config.device)
                    
                    # Classification
                    logits = classifier((x_pur+1)*0.5)

                    # Calculate loss
                    loss = F.cross_entropy(logits, y, reduction="sum")

                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                x_adv = x_adv.detach()

            grad /= eot_step
            grad = grad.sign()
            # x_adv = x_adv + args.num_steps * grad
            x_adv = x_adv + args.step_size_adv * grad

            delta = x_adv - X
            delta_norms = torch.norm(delta.view(X.shape[0], -1), p=2, dim=1)
            factor = args.epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            x_adv = X + delta
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, 0, 1.0)

        print(f'eta norm: {torch.norm((x_adv - X).view(X.shape[0], -1), p=2, dim=1).max()}')
        return x_adv

    def _BPDA_EOT_whitebox(model, cm_diffusion, classifier, X, y, mean, std, eot_step=20):
        from utils import diffusion_to_gaussian, purify_cifar, purify_imagenet
        x_adv = X.detach().clone().to(config.device)
        model.to(config.device)
        for _ in range(args.num_steps):
            grad = torch.zeros_like(x_adv)
            
            for _ in range(eot_step):
                with torch.no_grad():
                    # Purification module
                    noise, x_diff = diffusion_to_gaussian(x_adv*2-1, args.atk_t_cm, config)

                    if args.domain in ['cifar10']:
                        # (-sigma, sigma) --> (-1, 1)
                        preprocessed_x = purify_cifar(model, x_diff, noise, config.device)
                    elif args.domain in ['imagenet100']:
                        preprocessed_x = purify_imagenet(x_diff, model, cm_diffusion, noise, config.device)
                preprocessed_x.requires_grad = True
                
                with torch.enable_grad():
                    # Classification
                    logits = classifier((preprocessed_x+1)*0.5)
                    
                    # Calculate loss
                    loss = F.cross_entropy(logits, y, reduction="sum")
                    
                    grad += torch.autograd.grad(loss, [preprocessed_x])[0].detach()
                x_adv = x_adv.detach()

            grad /= eot_step
            grad = grad.sign()
            # x_adv = x_adv + args.num_steps * grad
            x_adv = x_adv + args.step_size_adv * grad

            # Projection
            x_adv = X + torch.clamp(x_adv - X, min=-args.epsilon, max=args.epsilon)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, 0, 1.0)

        print(f'eta max: {(x_adv - X).max()}; eta min: {(x_adv - X).min()}')
        return x_adv
    
    def _AA_Attack_whitebox(model, cm_diffusion, classifier, X, y, mean, std, eot_step=None):
        from autoattack import AutoAttack
        import torch.nn as nn
        from utils import diffusion_to_gaussian, purify_cifar, purify_imagenet

        class Classifier_multisteppur(nn.Module):
            def __init__(self, model, cm_diffusion, classifier):
                super().__init__()
                self.model = model
                self.cm_diffusion = cm_diffusion
                self.base_classifier = classifier
            
            def purify(self, x):
                x = x.to(config.device)
                self.model.to(config.device)
                # Purification module
                noise, x_diff = diffusion_to_gaussian(x*2-1, args.atk_t_cm, config)

                if args.domain in ['cifar10']:
                    # (-sigma, sigma) --> (-1, 1)
                    x_pur = purify_cifar(model, x_diff, noise, config.device)
                elif args.domain in ['imagenet100']:
                    x_pur = purify_imagenet(x_diff, model, cm_diffusion, noise, config.device)
                
                return (x_pur+1)*0.5
            
            def forward(self, x):
                # with torch.enable_grad():
                x_pur = self.purify(x)
            
                return self.base_classifier(x_pur)

        classifier_multisteppur = Classifier_multisteppur(model, cm_diffusion, classifier)

        attack_version = args.attack_version  # ['standard', 'rand', 'custom', 'fast']
        if getattr(args, 'log_dir_adv', None):
            if args.log_dir_adv !="":
                log_dir_adv = args.log_dir_adv
                log_path=f'{log_dir_adv}/log_sde_adv.txt'
                verbose = True
            else:
                log_path = None
                verbose = False				
        else:
            log_path = None
            verbose = False
        adversary_resnet = AutoAttack(classifier_multisteppur, norm='Linf', eps=args.epsilon,
                                  version=attack_version,log_path=log_path, verbose=verbose)
        adversary_resnet.apgd.eot_iter = 20
        
        x_adv = adversary_resnet.run_standard_evaluation(X, y, bs=X.shape[0])

        print(f'eta max: {(x_adv - X).max()}; eta min: {(x_adv - X).min()}')
        return x_adv
    
    def _AA_Attack_L2_whitebox(model, cm_diffusion, classifier, X, y, mean, std, eot_step=None):
        from autoattack import AutoAttack
        import torch.nn as nn
        from utils import diffusion_to_gaussian, purify_cifar, purify_imagenet

        class Classifier_multisteppur(nn.Module):
            def __init__(self, model, cm_diffusion, classifier):
                super().__init__()
                self.model = model
                self.cm_diffusion = cm_diffusion
                self.base_classifier = classifier
            
            def purify(self, x):
                x = x.to(config.device)
                self.model.to(config.device)

                # Purification module
                noise, x_diff = diffusion_to_gaussian(x*2-1, args.atk_t_cm, config)

                if args.domain in ['cifar10']:
                    # (-sigma, sigma) --> (-1, 1)
                    x_pur = purify_cifar(model, x_diff, noise, config.device)
                elif args.domain in ['imagenet100']:
                    x_pur = purify_imagenet(x_diff, model, cm_diffusion, noise, config.device)
                
                return (x_pur+1)*0.5
            
            def forward(self, x):
                # with torch.enable_grad():
                x_pur = self.purify(x)
            
                return self.base_classifier(x_pur)

        classifier_multisteppur = Classifier_multisteppur(model, cm_diffusion, classifier)

        attack_version = args.attack_version  # ['standard', 'rand', 'custom', 'fast']
        if getattr(args, 'log_dir_adv', None):
            if args.log_dir_adv !="":
                log_dir_adv = args.log_dir_adv
                log_path=f'{log_dir_adv}/log_sde_adv.txt'
                verbose = True
            else:
                log_path = None
                verbose = False				
        else:
            log_path = None
            verbose = False
        adversary_resnet = AutoAttack(classifier_multisteppur, norm='L2', eps=args.epsilon,
                                  version=attack_version,log_path=log_path, verbose=verbose)
        
        x_adv = adversary_resnet.run_standard_evaluation(X, y, bs=X.shape[0])

        print(f'eta norm: {torch.norm((x_adv - X).view(X.shape[0], -1), p=2, dim=1).max()}')
        return x_adv


    stack_kernel = gaussian_kernel().cuda()
    is_transferred = True if (attack_classifier is not None and attack_classifier != classifier) else False
    classifier.eval()

    if is_transferred:
        attack_classifier.eval()
    else:
        attack_classifier = classifier

    with torch.no_grad():
        top1, top5, num_data = 0, 0, 0

        input = input.cuda(non_blocking=True).mul_(std).add_(mean)
        target = target.cuda(non_blocking=True)

        assert attack_method in ['PGD_EOT','PGD_EOT_L2','BPDA_EOT','AA_Attack','AA_Attack_L2']

        X_adv = eval('_{}_whitebox'.format(attack_method))(model, cm_diffusion, classifier, input, target, mean, std)

        outputs = classifier(X_adv.sub(mean).div(std))
        prec1, prec5 = accuracy(outputs, target, topk=(1, 5))

        outputs = outputs.max(dim=1)[1]
        correct_batch = target.eq(outputs)
        top1 = torch.sum(correct_batch).item()/target.size(0)

    return X_adv, top1, prec5/100.