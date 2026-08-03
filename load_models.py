import torch
import torch.nn as nn
import torchvision.models as models

from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib
from improved_diffusion.script_util import create_model_and_diffusion as create_model_and_diffusion_imagenet100
from improved_diffusion.script_util import  model_and_diffusion_defaults as model_and_diffusion_defaults_imagenet100

from robustbench import load_model
from utils import dict2namespace, restore_checkpoint

def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict

# load image classifiers
def get_image_classifier(classifier_name):
    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

        def forward(self, x):
            x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
            return self.resnet(x)

    if 'imagenet100' in classifier_name:
        if 'resnet50' in classifier_name:
            print('using imagenet100 resnet50...')
            model = models.resnet50(num_classes=100).eval()
            model_path = 'pretrained/clf/imagenet/resnet50-64_model_best.pth.tar'
            model.load_state_dict(torch.load(model_path)['state_dict'])
        elif 'resnet101' in classifier_name:
            print('using imagenet100 resnet101...')
            model = models.resnet101(num_classes=100).eval()
            model_path = 'pretrained/clf/imagenet/resnet101-64_model_best.pth.tar'
            model.load_state_dict(torch.load(model_path)['state_dict'])
        elif 'wideresnet-50-2' in classifier_name:
            print('using imagenet100 wideresnet-50-2...')
            model = models.wide_resnet50_2(num_classes=100).eval()
            model_path = 'pretrained/clf/imagenet/wideresnet-50-2-64_model_best.pth.tar'
            model.load_state_dict(torch.load(model_path)['state_dict'])
        
        wrapper_resnet = _Wrapper_ResNet(model)
    
    elif 'cifar10' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            print('using cifar10 wideresnet-28-10...')
            model_path = 'pretrained/clf'
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf', model_dir=model_path)  # pixel in [0, 1]

        elif 'wideresnet-70-16' in classifier_name:
            print('using cifar10 wideresnet-70-16...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            model_path = 'pretrained/clf/cifar10/weights-best.pt'
            print(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            print(f"=> loaded wideresnet-70-16 checkpoint")

        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = model
    
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    return wrapper_resnet


# load diffusion models
def get_diffusion_model(domain, config):
    if domain in ['cifar10']:
        # config = dict2namespace(config)
        diffusion = mutils.create_model(config)
        optimizer = get_optimizer(config, diffusion.parameters())
        ema = ExponentialMovingAverage(
            diffusion.parameters(), decay=config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=diffusion, ema=ema)
        restore_checkpoint('pretrained/diffusion/score_sde/checkpoint_8.pth', state, config.device)
        ema.copy_to(diffusion.parameters())
        diffusion.eval().to(config.device)
    elif domain in ['imagenet100']:
        # config = dict2namespace(config)
        model_config = model_and_diffusion_defaults_imagenet100()
        model_config.update(vars(config.model))
        diffusion, _ = create_model_and_diffusion_imagenet100(**model_config)
        diffusion.load_state_dict(torch.load('pretrained/diffusion/guided_diffusion/imagenet100_uncond_vlb_100M_1500K.pt', map_location='cpu'))
        diffusion.eval().to(config.device)
    
    return diffusion

# load consistency models
defaults = dict(
    training_mode="edm",
    generator="determ",
    clip_denoised=True,
    num_samples=10000,
    batch_size=16,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    steps=40,
    model_path="",
    seed=42,
    ts="",
)
model_and_diffusion_defaults =dict(
    sigma_min=0.002,
    sigma_max=80.0,
    image_size=64,
    num_channels=128,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    num_head_channels=-1,
    attention_resolutions="32,16,8",
    channel_mult="",
    dropout=0.0,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    learn_sigma=False,
    weight_schedule="karras"
)
args_imagenet64 = dict(
    batch_size=64,
    training_mode="consistency_distillation",
    sampler="onestep",
    model_path="pretrained/cm/imagenet/model200000.pt",
    attention_resolutions="32,16,8",
    class_cond=False,
    use_scale_shift_norm=True,
    dropout=0.0,
    image_size=64,
    num_channels=192,
    num_head_channels=64,
    num_res_blocks=3,
    num_samples=500,
    resblock_updown=True,
    use_fp16=True,
    weight_schedule="uniform"
)

def get_consistency_model(domain):
    if domain in ['imagenet100']:
        from cm_imagenet.cm.script_util import (
            create_model_and_diffusion,
            update_dict,
        )

        model, diffusion = create_model_and_diffusion(
            **update_dict(args_imagenet64, model_and_diffusion_defaults),
            distillation=True,
        )
        model.load_state_dict(torch.load(args_imagenet64['model_path'], map_location="cpu"))

        if args_imagenet64['use_fp16']:
            model.convert_to_fp16()
        
        return model, diffusion
    elif domain in ['cifar10']:
        from cm_cifar10.jcm.models import utils as mutils
        from cm_cifar10.configs.cifar10_ve_cd import get_config

        config_cm = get_config()
        model = mutils.create_model(config_cm)
        model.load_state_dict(torch.load('pretrained/cm/cifar10/cd_lpips_checkpoint.pth'))

        return model