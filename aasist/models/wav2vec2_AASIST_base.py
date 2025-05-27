"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import math
# from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from peft import LoraConfig
from peft import get_peft_model


class ResidualUnit(nn.Module):
    

    def __init__(self,n_samples_in, n_samples_out,n_filters_in, n_filters_out,
                 dropout_keep_prob=0.2, kernel_size=17):
        
        super(ResidualUnit, self).__init__()
        self.n_samples_in = n_samples_in
        self.n_samples_out = n_samples_out
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.downsample = self.n_samples_in // self.n_samples_out
        
        
            
        self.dropout_rate = dropout_keep_prob
        self.kernel_size = kernel_size
        
        
        self.bn0 = nn.BatchNorm1d(self.n_filters_out)
        self.drop = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()
        
        # skip connection
        if self.downsample >= 1 :                                                          # >
            self.max1d = nn.MaxPool1d(self.downsample, stride=self.downsample, padding=0)  # 0,1
        else :
            raise ValueError("Number of samples should always decrease.")
        self.conv1d_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, 1,bias=False)
        
        self.conv1d_1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, self.kernel_size,padding=8,bias=False)
        self.conv1d_2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, self.kernel_size,padding=8,stride = self.downsample,bias=False)
        

    def forward(self, inputs):
        """Residual unit."""
        x, y = inputs
        y = self.max1d(y)
        y = self.conv1d_skip(y)
        x = self.conv1d_1(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv1d_2(x)
        x = torch.add(x,y)
        y = x
        x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
    
        return (x, y)

class dnn_model(nn.Module):
    def __init__(self, classes_num, input_len):
        
        super(dnn_model, self).__init__() 
        '''
        input shape : (B,1,1250)
        input_len : 160, 192, 288, 224, 320, 
        '''
        
        self.classes_num = classes_num
        self.input_len = input_len
        self.kernel_size = 17
        self.conv = nn.Conv1d(1, 64, self.kernel_size,bias=False,padding=8)  # , padding='same'
        self.bn = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        self.fc_reshape = nn.Linear(self.input_len,512)
        self.residual1 = ResidualUnit(512,128,64,128, kernel_size=self.kernel_size)
        self.residual2 = ResidualUnit(128,32,128,196, kernel_size=self.kernel_size)
        self.residual3 = ResidualUnit(32,16,196,256, kernel_size=self.kernel_size)
        self.residual4 = ResidualUnit(16,8,256,320, kernel_size=self.kernel_size)
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(nn.Linear(8*320, 512),
                                nn.Linear(512, 128),
                                nn.Linear(128, 32),
                                nn.Linear(32, self.classes_num)
                                )
        

    def forward(self, input):
        """
        Input: (batch_size, channel, data_length)"""

        input = input.float()
        x = input.unsqueeze(1)
#        print('input : ', input.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_reshape(x)
        x,y = self.residual1((x,x))
        x,y = self.residual2((x,y))
        x,y = self.residual3((x,y))
        x,_ = self.residual4((x,y))
        x = self.flatten(x)
        out = self.fc(x)
       
        
        return out

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # att_map = torch.matmul(att_map, self.att_weight12)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)

        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        self.filt1 = nb_filts[0]
        self.filt2 = nb_filts[1]
        
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(3, 2),    #(2, 3)
                               padding=(1, 1),        #(1, 1)
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(3, 2),     #(2, 3)
                               padding=(1, 0),         #(0, 1)
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(1, 0),       #(0, 1)
                                             kernel_size=(3, 1),   #(1, 3)
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((3, 2))   # 내가 추가
 #       self.mp = nn.MaxPool2d((2, 1))  # self.mp = nn.MaxPool2d((1,4))  # self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

#        print('res_block_out1: ',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
#        print('res_block_out2: ',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)
#            print('identity_out: ',identity.shape)
        out += identity
        if self.filt1 == self.filt2 :               # 내가 추가
            out = self.mp(out)
#        else :
#            out = self.mp(out)
        return out


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
      
        self.d_args = d_args
        filts = d_args["filts"]                 # "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        gat_dims = d_args["gat_dims"]           #  "gat_dims": [64, 32],
        pool_ratios = d_args["pool_ratios"]     # "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        temperatures = d_args["temperatures"]   # "temperatures": [2.0, 2.0, 100.0, 100.0],
        
        total_transformer_layers = d_args["total_transformer_layers"]
        n_frz_layers = d_args["n_frz_layers"]          # -1 : 모든 Transformer Layer.requires_grad = True , n : n Transformer Layer까지 frz.
        version = d_args["version"]
        w2v2 = Wav2Vec2ForCTC.from_pretrained(version,num_hidden_layers=total_transformer_layers)
        
        self.w2v2 = layer_freeze(w2v2, total_transformer_layers, n_frz_layers)
        
        self._init_weights(w2v2) # closed condition


#        self.conv_time = CONV(out_channels=filts[0],
#                              kernel_size=d_args["first_conv"],   #128
#                              in_channels=1)
#        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),    # 123444
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.pos_S = nn.Parameter(torch.randn(1, 15, filts[-1][-1]))   #nn.Parameter(torch.randn(1, 23, filts[-1][-1])) ---300m:12, 1b:15
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        
    def _init_weights(self, model):
        """
        Initialize the model's weights randomly.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.constant_(module.weight[module.padding_idx], 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, Freq_aug=False):
        '''
        input : 약 4초 64600
        '''
#        print('w2v2 start')
#        print('input : ', x.shape)   
#        x = x.unsqueeze(1)
#        x = self.conv_time(x, mask=Freq_aug)   # output : (70,Length), length : 64,---
        
        
        w2v2_out = self.w2v2(x).logits   
        w2v2_out = torch.transpose(w2v2_out,1,2)    # output : (1280,time)
#        print('w2v2 : ', w2v2_out.shape)
        
        x = w2v2_out.unsqueeze(dim=1)
#        x = F.max_pool2d(torch.abs(x), (3, 3))
#        x = self.first_bn(x)
#        x = self.selu(x)
#        print('1st maxpool & encoder input : ', x.shape)

        # get embeddings using encoder
        # (#bs, #filt, #spec, #seq)
        e = self.encoder(x)
#        print('encoder out : ', e.shape)

        # spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
#        print('1st gat_outS : ',gat_S.shape)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
#        print('1st graph pool_outS : ',out_S.shape)

        # temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
#        print('1st gat_outT : ',gat_T.shape)
        out_T = self.pool_T(gat_T)
#        print('1st graph pool_outT : ',out_T.shape)
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
#        print('master dim : ',master1.shape)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)
#        print('HtrgGAT outT1 : ',out_T1.shape)
#        print('HtrgGAT outS1 : ',out_S1.shape)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output
    
    def get_nb_trainable_parameters(self):# -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self,return_option=False):# -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        if return_option :
            return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        else :
            print(
                f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
            )

    

    

    


def layer_freeze(w2v2, total_layer,frz_layer) :
    w2v2.dropout = nn.Identity()
    w2v2.lm_head = nn.Identity()
    for para in w2v2.parameters():
        para.requires_grad = False

    for name ,child in w2v2.named_children():
        if name == 'wav2vec2':
            for nam,chil in child.named_children():
                if nam == 'encoder' :
                    for na,chi in chil.named_children():
                        if na == 'layers' :
                            for i in range(total_layer):
                                if frz_layer == -1 :
                                    for para in chi.parameters():
                                        para.requires_grad = True
                                elif frz_layer <= i :
                                    for k,para in enumerate(chi[i].parameters()):
                                        para.requires_grad = True
    return w2v2                                                 

class W2V2_SE_Res2Net_frz_xls_r_1b(nn.Module):

    def __init__(self):

        super(W2V2_SE_Res2Net_frz_xls_r_1b, self).__init__()
                                                 
        total_transformer_layers = 12
        n_frz_layers = -1                              # -1 : 모든 Transformer Layer.requires_grad = True , n : n Transformer Layer까지 frz.
        res2net_layers = 2
        
        self.w2v2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base",num_hidden_layers=total_transformer_layers)
                                                 
        self.w2v2 = layer_freeze(self.w2v2, total_transformer_layers, n_frz_layers)
                                                 

    def forward(self, x):
                                                 
        w2v2_out = self.w2v2(x).logits   
        w2v2_out = torch.transpose(w2v2_out,1,2)    # output : (1280,time)
                                          
        
        return out

