from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.commons.hparams import hparams
from torchdyn.core import NeuralODE
sigma = 1e-4

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class Wrapper(nn.Module):
    def __init__(self, net, cond, num_timesteps, dyn_clip):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond
        self.num_timesteps = num_timesteps
        self.dyn_clip = dyn_clip

    def forward(self, t, x, args):
        t = torch.tensor([t * self.num_timesteps], device=t.device).long()
        ut = self.net.denoise_fn(x, t, self.cond)
        if hparams['f0_sample_clip']:
            x_recon = (1 - t / self.num_timesteps) * ut + x
            if self.dyn_clip is not None:
                x_recon.clamp_(self.dyn_clip[0].unsqueeze(1), self.dyn_clip[1].unsqueeze(1))
            else:
                x_recon.clamp_(-1., 1.)
            ut = (x_recon - x) / (1 - t / self.num_timesteps)
        return ut

class ReflowF0(nn.Module):
    def __init__(self, out_dims, denoise_fn, timesteps=1000, f0_K_step=1000, loss_type='l1'):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims
        self.K_step = f0_K_step
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

    def q_sample(self, x_start, t, noise=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x1 = x_start
        x0 = noise
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).unsqueeze(1).float() / self.num_timesteps

        if hparams['flow_qsample'] == 'sig':
            epsilon = torch.randn_like(x0)
            xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0 + sigma * epsilon
        else:
            xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0 
        return xt

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        xt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x1 = x_start
        x0 = noise
        
        v_pred = self.denoise_fn(xt, t, cond)
        ut = x1 - x0 
        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((ut - v_pred).abs() * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
            else:
                loss = ((ut - v_pred).abs()).mean()
        elif self.loss_type == 'l2':
            if nonpadding is not None:
                loss = (F.mse_loss(ut, v_pred,  reduction='none') * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
            else:
                loss_simple = F.mse_loss(ut, v_pred,  reduction='none')
                loss = torch.mean(loss_simple)
        else:
            raise NotImplementedError()
        
        return loss

    def forward(self, cond, f0=None, nonpadding=None, ret=None, infer=False,dyn_clip=None,solver='euler'):
        b = cond.shape[0]
        device = cond.device
        if not infer:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            x = f0.unsqueeze(1).unsqueeze(1)# [B, 1, M, T]
            return self.p_losses(x, t, cond, nonpadding=nonpadding)
        else:
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x0 = torch.randn(shape, device=device)
            neural_ode = NeuralODE(self.ode_wrapper(cond, self.num_timesteps, dyn_clip), solver=solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            t_span = torch.linspace(0, 1, self.K_step + 1)
            eval_points, traj = neural_ode(x0, t_span)
            x = traj[-1]
            x = x[:, 0].transpose(1, 2)
        return x

    def ode_wrapper(self, cond, num_timesteps, dyn_clip):
        return Wrapper(self, cond, num_timesteps, dyn_clip)