import torch
import logging
import pickle
import glob
import random
import numpy as np
from .model_utils import *
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Union
from functools import partial


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def alpha_bar_cosine(time_step, s = 0.0, e = 1.0, tau=1.0, clip_min = 1e-9, **kwargs):
    v_start = np.cos((s + 0.008) / 1.008 * np.pi / 2) ** (2*tau)
    v_end = np.cos((e + 0.008) / 1.008 * np.pi / 2) ** (2*tau)
    v = np.cos((time_step * (e-s)+s + 0.008) / 1.008 * np.pi / 2) ** (2*tau)
    return np.clip((v_end - v) / (v_end - v_start), clip_min, 1)
    # return np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2
    
def alpha_bar_log(time_step, tau = 2.5, clip_min = 1e-9, **kwargs):
    # ! Hard code to shift the schedule 
    # delta = 10**-tau
    # return np.clip(np.log(time_step+delta)/np.log(delta), clip_min, 1)
    
    delta = 10**-tau
    v_start = np.log(delta)
    v_end = np.log(1+delta)
    v = np.log(time_step+delta)
    return np.clip((v_end - v) / (v_end - v_start), clip_min, 1)

def alpha_bar_linear(time_step, clip_min = 1e-9, **kwargs):
    return np.clip((1-time_step), clip_min, 1.)

def alpha_bar_sigmoid(time_step, s = -3, e = 3, tau=1, clip_min = 1e-9, **kwargs):
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    v_start = sigmoid(s/tau)
    v_end = sigmoid(e/tau)
    v = sigmoid((time_step*(e-s)+s)/tau)
    return np.clip((v_end - v) / (v_end - v_start), clip_min, 1)

def get_beta_schedule(variant, num_diffusion_timesteps, clip_min = 1e-9, max_beta=0.999, scale = 1.0, **kwargs):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    """

    if variant == 'cosine':
        alpha_bar = partial(alpha_bar_cosine, clip_min = clip_min, **kwargs)
    elif variant == 'log':
        alpha_bar = partial(alpha_bar_log, clip_min = clip_min, **kwargs)
    elif variant == 'sigmoid':
        alpha_bar = partial(alpha_bar_sigmoid, clip_min = clip_min, **kwargs)
    elif variant == 'linear':
        alpha_bar = partial(alpha_bar_linear, clip_min = clip_min, **kwargs)

    betas = [(1 - alpha_bar(time_step = 0)*scale)]
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(time_step = t2) / alpha_bar(time_step = t1), max_beta))
    # betas[0] = betas[0]*(1-scale)
    return torch.tensor(betas, dtype=torch.float32)

class DDPM_Sampler(torch.nn.Module):
    def __init__(
        self, steps=100, schedule='cosine',
        clamp_val: float = 5.0, **kwargs
    ):
        super().__init__()
        self.num_steps = steps
        self.schedule = schedule
        self.clamp_val = clamp_val
        
        betas = get_beta_schedule(variant = self.schedule, num_diffusion_timesteps = self.num_steps, **kwargs)
        betas_sqrt = betas.sqrt()
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        
        self.register_buffer('betas', betas[1:])
        self.register_buffer('betas_sqrt', betas_sqrt[1:])
        self.register_buffer('alphas', alphas[1:])
        self.register_buffer('alphas_cumprod', alphas_cumprod[1:])
        
    @torch.no_grad()
    def add_noise(
        self, 
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor
    ):

        assert (timesteps < self.num_steps).all()
        
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)
            
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        noised_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noised_samples
    
    def get_noise(
        self,
        x_0: torch.FloatTensor,
        x_t: torch.FloatTensor,
        timesteps: Union[int, torch.IntTensor],
        gt_noise: torch.FloatTensor = None
    ):
        assert (timesteps < self.num_steps).all()
        
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=x_0.device, dtype=x_0.dtype)
        timesteps = timesteps.to(x_0.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(x_0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(x_0.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # noised_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        noise = (x_t - sqrt_alpha_prod * x_0) / sqrt_one_minus_alpha_prod
        
        if gt_noise is not None:
            scaled_error = torch.nn.functional.mse_loss(
                x_t - sqrt_alpha_prod * x_0, gt_noise*sqrt_one_minus_alpha_prod, reduction='mean')
        else:
            scaled_error = None
        
        return noise, scaled_error
    
    def set_timesteps(
        self,
        num_inference_steps = None,
        device = None):

        timesteps = (
            np.linspace(0, self.num_steps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timesteps: Union[int, torch.IntTensor],
        sample: torch.FloatTensor,
        prediction_type: str = "sample"
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        if not isinstance(timesteps, int):
            while len(timesteps.shape) < len(model_output.shape):
                timesteps = timesteps.unsqueeze(-1)
        # Compute predicted previous sample µ_t-1
        pred_prev_sample_mean = self.q_mean(model_output, timesteps, sample, prediction_type=prediction_type)
        # 6. Add noise
        device = model_output.device
        variance_noise = torch.randn(model_output.shape, device=device, dtype=model_output.dtype)
        
        variance = (self.q_variance(timesteps)**0.5) * variance_noise

        pred_prev_sample = pred_prev_sample_mean + variance

        return pred_prev_sample
    
    def q_mean(self,
        model_output: torch.FloatTensor,
        timesteps: Union[int, torch.IntTensor],
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        # if type(timestep) == int:
        #     t = timestep
        # else:
        #     t = timestep[0][0]
        prev_t = timesteps - 1
        if isinstance(prev_t, int):
            prev_t = max(prev_t, 0)
        else:
            prev_t = prev_t.clamp(min=0)
            while len(timesteps.shape) < len(model_output.shape):
                timesteps = timesteps.unsqueeze(-1)
                prev_t = prev_t.unsqueeze(-1)
        

        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timesteps]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] #if prev_t >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Compute predicted original sample from predicted noise also called "predicted x_0"
        if prediction_type == "sample" or prediction_type == "mean":
            pred_original_sample = model_output
        elif prediction_type == "error":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif prediction_type == "v":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise NotImplementedError

        # 3. Clip or threshold "predicted x_0"
        pred_original_sample = pred_original_sample.clamp(-self.clamp_val, self.clamp_val)
        # samxple = sample.clamp(-self.clamp_val, self.clamp_val)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t 

        # 5. Compute predicted previous sample µ_t
        pred_prev_sample_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        return pred_prev_sample_mean 
    
    def q_x0(
        self,
        model_output: torch.FloatTensor,
        timesteps: Union[int, torch.IntTensor],
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
    ):
        """
        Predict the denoised x0 from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        

        # 2. Compute predicted original sample from predicted noise also called "predicted x_0"
        if prediction_type == "sample" or prediction_type == "mean":
            pred_original_sample = model_output
        elif prediction_type == "error":
            alpha_prod_t = self.alphas_cumprod[timesteps]
            for _ in range(len(sample.shape)-len(alpha_prod_t.shape)):
                alpha_prod_t = alpha_prod_t[..., None]
            beta_prod_t = 1 - alpha_prod_t
            
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # elif prediction_type == "v":
        #     pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise NotImplementedError

        return pred_original_sample

    def q_variance(self, timesteps):
        # if t == 0:
        #     return 0
        prev_t = timesteps - 1
        if isinstance(prev_t, int):
            prev_t = max(prev_t, 0)
        else:
            prev_t = prev_t.clamp(min=0)
        alpha_prod_t = self.alphas_cumprod[timesteps]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        variance = beta_prod_t_prev / beta_prod_t * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance
        
class DDIM_Sampler(DDPM_Sampler):
    def __init__(self, steps=100, schedule='cosine', clamp_val: float = 5.0):
        super().__init__(steps, schedule, clamp_val)
        
    def copy_from_ddpm(self, ddpm: DDPM_Sampler):
        self.num_steps = ddpm.num_steps
        self.schedule = ddpm.schedule
        self.clamp_val = ddpm.clamp_val
        
        self.register_buffer('betas', ddpm.betas.clone())
        self.register_buffer('betas_sqrt', ddpm.betas_sqrt.clone())
        self.register_buffer('alphas', ddpm.alphas.clone())
        self.register_buffer('alphas_cumprod', ddpm.alphas_cumprod.clone())
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timesteps: Union[int, torch.IntTensor],
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
        eta: float = 0.0
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        if not isinstance(timesteps, int):
            while len(timesteps.shape) < len(model_output.shape):
                timesteps = timesteps.unsqueeze(-1)
        # Compute predicted previous sample µ_t-1
        pred_prev_sample_mean = self.q_mean(model_output, timesteps, sample, prediction_type=prediction_type, eta=eta)
        
        return pred_prev_sample_mean
    
    def q_variance(self, timestep, prev_timestep = None):
        if prev_timestep is None:
            prev_timestep = timestep - 1
        if isinstance(prev_timestep, int):
            prev_timestep = max(prev_timestep, 0)
        else:
            prev_timestep = prev_timestep.clamp(min=0)
            
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        
        return variance

    def q_mean(self,
        model_output: torch.FloatTensor,
        timesteps: Union[int, torch.IntTensor],
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
        eta = 0.0
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        prev_t = timesteps - 1
        if isinstance(prev_t, int):
            prev_t = max(prev_t, 0)
        else:
            prev_t = prev_t.clamp(min=0)
            while len(timesteps.shape) < len(model_output.shape):
                timesteps = timesteps.unsqueeze(-1)
                prev_t = prev_t.unsqueeze(-1)

        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timesteps]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]

        beta_prod_t = 1 - alpha_prod_t
        # beta_prod_t_prev = 1 - alpha_prod_t_prev
        # current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        # current_beta_t = 1 - current_alpha_t

        # 2. Compute predicted original sample from predicted noise also called "predicted x_0"
        if prediction_type == "sample" or prediction_type == "mean":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif prediction_type == "error":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif prediction_type == "v":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise NotImplementedError

        # 3. Clip or threshold "predicted x_0"
        pred_original_sample = pred_original_sample.clamp(-self.clamp_val, self.clamp_val)
        # pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5)
        # current_sample_coeff = (1 - current_alpha_t) ** 0.5
        
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.q_variance(timesteps, prev_t)
        std_dev_t = eta * variance ** (0.5)
        
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon


        # 5. Compute predicted previous sample µ_t
        pred_prev_sample_mean = pred_original_sample_coeff * pred_original_sample + pred_sample_direction
        if eta > 0.0:
            noise = torch.randn(model_output.shape, device=model_output.device, dtype=model_output.dtype)
            pred_prev_sample_mean += std_dev_t* noise
            
        return pred_prev_sample_mean
