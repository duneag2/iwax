import sys
import os
import pickle
from tqdm import tqdm
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import soundfile as sf

import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, kernel_size, low_hz, high_hz, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv_fast,self).__init__()

        if in_channels != 1: 
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.low_hz = np.float64(low_hz)
        self.high_hz = np.float64(high_hz)

        hz = np.array([self.low_hz, self.high_hz])

        # filter lower frequency (1, 1)
        self.low_hz_ = torch.Tensor(hz[:-1]).view(-1, 1)

        # filter frequency band (1, 1)
        self.band_hz_ = torch.Tensor(np.diff(hz)).view(-1, 1)

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.n_lin = n_lin
        
        # Hamming window
        # self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)  

        # Blackman window
        self.window_=0.42-0.5*torch.cos(2*math.pi*n_lin/self.kernel_size)+0.08*torch.cos(4*math.pi*n_lin/self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    
        ############### useless codes, just to visualize the filter #################
        low = self.low_hz_
        high = torch.clamp(low + torch.abs(self.band_hz_), 0, self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        self.band_pass_center = band_pass_center
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        
        band_pass = band_pass / (2*band[:,None])
        
        self.filter = (band_pass).view(1, 1, self.kernel_size)
        ###########################################################################
        
    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, 1, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.low_hz_
        high = torch.clamp(low + torch.abs(self.band_hz_), 0, self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)

        self.band_pass_center = band_pass_center
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        
        band_pass = band_pass / (2*band[:,None])
        
        self.filter = (band_pass).view(1, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filter, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
        

def compute_coeff(low_freq, high_freq):
    start_time = 0
    end_time = 1
    sample_rate = 16000
    time_range = np.arange(start_time, end_time, 1/sample_rate)
    theta = 0
    amplitude = 1

    mid_freq = (low_freq + high_freq) / 2
    sine = amplitude * np.sin(2 * np.pi * mid_freq * time_range + theta)


    kernel_size = 513
    padding = kernel_size // 2  
    sinc_temp = SincConv_fast(kernel_size=kernel_size, low_hz=low_freq, high_hz=high_freq, padding=padding)

    sine_tensor = torch.Tensor(sine).view(1,1,-1)
    filtered_output = sinc_temp(sine_tensor)
    filtered_array = filtered_output.view(-1).numpy()
    normalization_coeff = filtered_array.max()

    return normalization_coeff
