"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午10:50
@Author  : Yang "Jan" Xiao 
@Description : train_utils
"""
import torch
import torch_optimizer
from torch import optim
from torch import nn
from torchaudio.transforms import MFCC, MelSpectrogram
from networks.bcresnet import BCResNet
from networks.tcresnet import TCResNet
from networks.matchboxnet import MatchboxNet
from networks.matchbox.matchbox_model import EncDecBaseModel
from networks.kwt import kwt_from_name
from networks.convmixer import KWSConvMixer
from networks.mhatt_rnn import MHAtt_RNN
from networks.kwm.aum_model import AUMModel
from networks.kwm.ast_model import ASTModel
from networks.kwm.kwm import KWM, kwm_from_name
from networks.kwm.kwmt import KWMT, kwmt_from_name
from networks.kwm.utils import WarmUpLR
from transformers import get_polynomial_decay_schedule_with_warmup

class MFCC_KWS_Model(nn.Module):
    def __init__(self, model) -> None:
        super(MFCC_KWS_Model,self).__init__()
        self.mfcc = MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 480, "win_length": 480, "hop_length": 160, "n_mels": 80, "center": False, "f_min": 20, 'f_max': 7600},
        )
        self.model = model
    def forward(self, x):
        x = self.mfcc(x)
        x = self.model(x)
        return x


def select_optimizer(opt_name, lr, model, sched_name, wd, total_iters_s, total_iters_w, warmup):
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    elif opt_name == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-3)
    elif opt_name == "NovoGrad":
        opt = torch_optimizer.NovoGrad(model.parameters(), lr=lr, betas=(0.95, 0.5), weight_decay=0.001)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "adamw":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")

    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters_s, eta_min=1e-8)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, list(range(5, 70, 4)), gamma=0.85)
    elif sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    elif sched_name == "linear":
        scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=lr*10, end_factor=wd*10, total_iters=135)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "coswarm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=1e-8)
    elif sched_name == "poly":
        scheduler = get_polynomial_decay_schedule_with_warmup(opt, 
                    total_iters_w, total_iters_s, lr_end=1e-3, power=1.0, last_epoch=- 1)
    elif sched_name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999511)
    else:
        raise NotImplementedError("Please select the sched_name [cos, anneal, multistep]")
        # scheduler = None

    if warmup != 0:
        warmup_sche = WarmUpLR(opt, total_iters_w)
    else:
        warmup_sche = None
    
    schedulers = {
        "scheduler": None,
        "warmup": None
    }
    schedulers["scheduler"] = scheduler
    schedulers["warmup"] = warmup_sche

    return opt, schedulers


def select_model(model_name, total_class_num=None):
    # load the model.
    config = {
        "tcresnet8": [16, 24, 32, 48],
        "tcresnet14": [16, 24, 24, 32, 32, 48, 48],
        "tcresnet8-1.5": [24, 36, 48, 72],
        "tcresnet14-1.5": [24, 36, 36, 48, 48, 72, 72]
    }

    if "tcresnet" in model_name:
        model_name = model_name.split("_")[0]
        model = TCResNet(bins=40, n_channels=[int(cha * 1) for cha in config[model_name]],
                              n_class=total_class_num)
    elif "bcresnet" in model_name:
        model_name = model_name.split("_")[0]
        scale = int(model_name[-1])
        model = BCResNet(n_class=total_class_num, scale=scale)
    elif "matchboxnet" in model_name:
        model_name = model_name.split("_")[0]
        # b, r, c = model_name.split("-")[1:]
        # model = MatchboxNet(B=int(b), R=int(r), C=int(c), bins=64, kernel_sizes=None,num_classes=total_class_num)
        model = EncDecBaseModel(num_mels=64, final_filter=128, num_classes=total_class_num)
    elif "convmixer" in model_name:
        model = KWSConvMixer(input_size=[98, 64],num_classes=total_class_num)
    elif "rnn" in model_name:
        model_name = model_name.split("_")[0]
        n_head = model_name.split("-")[-1]
        model = MHAtt_RNN(num_classes=total_class_num, n_head=int(n_head))
    elif "kwt" in model_name:
        model_name = model_name.split("_")[0]
        model = kwt_from_name(model_name, total_class_num)
    elif "aum" in model_name:
        model = AUMModel()
    elif "ast" in model_name:
        model = ASTModel()
    elif "kwm" in model_name:
        # model = KWM()
        model_name = model_name.split("_")[0]
        parts = model_name.split("-")
        name = "-".join(parts[:-1])
        depth = parts[-1] 
        model = kwm_from_name(name, total_class_num, int(depth))
    # elif "kwmt" in model_name:
    #     model_name = model_name.split("_")[0]
    #     parts = model_name.split("-")
    #     name = "-".join(parts[:-1])
    #     depth = parts[-1] 
    #     model = kwmt_from_name(name, total_class_num, int(depth))
    else:
        model = None
    print(model)
    return model

if __name__ == "__main__":
    inputs = torch.randn(8, 1, 16000)
    # inputs = padding(inputs, 128)
    model = select_model("bcresnet2", 15)
    outputs = model(inputs)
    print(outputs.shape)
    print('num parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

