from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.commons.hparams import hparams
from torchdyn.core import NeuralODE

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class Wrapper(nn.Module):
    def __init__(self, net, cond, K_step):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond
        self.K_step = K_step
    def forward(self, t, x, args):
        t = torch.tensor([t * self.K_step] * x.shape[0], device=t.device).long()
        return self.net.denoise_fn(x, t, self.cond)


class Wrapper_CFG(nn.Module):
    def __init__(self, net, cond, ucond, cfg_scale, K_step):
        super(Wrapper_CFG, self).__init__()
        self.net = net
        self.cond = cond
        self.ucond = ucond
        self.cfg_scale = cfg_scale
        self.K_step = K_step

    def forward(self, t, x, args):
        t = torch.tensor([t * self.K_step] * x.shape[0], device=t.device).long()
        cond_in = torch.cat([self.ucond, self.cond])
        t_in = torch.cat([t] * 2)
        x_in = torch.cat([x] * 2)

        v_uncond, v_cond = self.net.denoise_fn(x_in, t_in, cond_in).chunk(2)
        v_out = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        
        return v_out

class FlowMel(nn.Module):
    def __init__(self, out_dims, denoise_fn, timesteps=1000, K_step=1000, loss_type=hparams.get('flow_loss_type', 'l1'), spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

    def forward(self, cond, gt_mels, coarse_mels, ret, infer, ucond, noise=None, cfg_scale=1.0, solver='euler'):
        b, *_, device = *cond.shape, cond.device

        cond = cond.transpose(1, 2)
        fs2_mels = coarse_mels

        if not infer:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            x = self.norm_spec(gt_mels)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            ret['flow'] = self.p_losses(x, t, cond)
        else:
            ucond = ucond.transpose(1, 2)
            x0 = noise
            if x0==None:
                x0 = default(noise, lambda: torch.randn_like(fs2_mels)).transpose(1, 2)[:, None, :, :]

            neural_ode = NeuralODE(self.ode_wrapper_cfg(cond, ucond, cfg_scale, self.num_timesteps), solver=solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            t_span = torch.linspace(0, 1, self.K_step + 1)
            eval_points, traj = neural_ode(x0, t_span)
            x = traj[-1]
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)
            ret['flow'] = 0.0
        return ret
        
    def ode_wrapper(self, cond, K_step):
        return Wrapper(self, cond, K_step)
    
    def ode_wrapper_cfg(self, cond, ucond, cfg_scale, K_step):
        return Wrapper_CFG(self, cond, ucond, cfg_scale, K_step)
            
    def q_sample(self, x_start, t, noise=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x1 = x_start
        x0 = noise
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0
        return xt
    
    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        # x_start: x1 (x0 in sd3), data point
        # t: discrete step
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

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

