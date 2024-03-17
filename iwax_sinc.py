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

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from utils import compute_coeff

parser = argparse.ArgumentParser(description='iwax')

parser.add_argument('--low_freq', required=True, help='low_freq')
parser.add_argument('--high_freq', required=True, help='high_freq')
parser.add_argument('--time', required=True, help='n/4')
parser.add_argument('--save_dir', required=True, help='where to save the XGBoost model')

args = parser.parse_args()

low_freq = int(args.low_freq)
high_freq = int(args.high_freq)
time = int(args.time)
save_dir = str(args.save_dir)

print('low_freq: ', low_freq)
print('high_freq: ', high_freq)
print('time: ', time)
print('save_dir: ', save_dir)

normalization_coeff = compute_coeff(low_freq=low_freq, high_freq=high_freq)

path_to_w2v2 = ''
sys.path.insert(0, path_to_w2v2)

w2v2 = torch.load('path/to/fine-tuned/w2v2')
model = torch.nn.Sequential(*list(w2v2.children())[:-17])


train_list_bonafide = []
train_list_spoof = []
with open("./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "r") as f:
    for line in f:
      if line.split()[-1] == 'bonafide':
        train_list_bonafide.append(line.split()[1]+'.flac')
      else:
        train_list_spoof.append(line.split()[1]+'.flac')


dev_list_bonafide = []
dev_list_spoof = []
with open("./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "r") as f:
    for line in f:
      if line.split()[-1] == 'bonafide':
        dev_list_bonafide.append(line.split()[1]+'.flac')
      else:
        dev_list_spoof.append(line.split()[1]+'.flac')


test_list_bonafide = []
test_list_spoof = []
with open("./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "r") as f:
    for line in f:
      if line.split()[-1] == 'bonafide':
        test_list_bonafide.append(line.split()[1]+'.flac')
      else:
        test_list_spoof.append(line.split()[1]+'.flac')

def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len==64600:
        return x
    elif x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

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
        
        band_pass = band_pass / (2*band[:,None]*normalization_coeff)
        
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
        
        band_pass = band_pass / (2*band[:,None]*normalization_coeff)
        
        self.filter = (band_pass).view(1, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filter, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
    

kernel_size = 513
padding = kernel_size // 2  
sinc = SincConv_fast(kernel_size=kernel_size, low_hz=low_freq, high_hz=high_freq, padding=padding)


train_feature_list_bonafide = []
train_feature_list_spoof = []

for i in tqdm(range(len(train_list_bonafide))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_train/flac/'+train_list_bonafide[i])
    sinc_filtered = sinc(torch.Tensor(pad_random(audio[0], 64600)).unsqueeze(dim=0))
    tensor = model(sinc_filtered.cuda())
    tensor2 = torch.squeeze(tensor.logits[:,int(time*len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    train_feature_list_bonafide.append(tensor2)

for i in tqdm(range(len(train_list_spoof))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_train/flac/'+train_list_spoof[i])
    sinc_filtered = sinc(torch.Tensor(pad_random(audio[0], 64600)).unsqueeze(dim=0))
    tensor = model(sinc_filtered.cuda())
    tensor2 = torch.squeeze(tensor.logits[:,int(time*len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    train_feature_list_spoof.append(tensor2)


dev_feature_list_bonafide = []
dev_feature_list_spoof = []

for i in tqdm(range(len(dev_list_bonafide))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_dev/flac/'+dev_list_bonafide[i])
    sinc_filtered = sinc(torch.Tensor(pad_random(audio[0], 64600)).unsqueeze(dim=0))
    tensor = model(sinc_filtered.cuda())
    tensor2 = torch.squeeze(tensor.logits[:,int(time*len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    dev_feature_list_bonafide.append(tensor2)

for i in tqdm(range(len(dev_list_spoof))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_dev/flac/'+dev_list_spoof[i])
    sinc_filtered = sinc(torch.Tensor(pad_random(audio[0], 64600)).unsqueeze(dim=0))
    tensor = model(sinc_filtered.cuda())
    tensor2 = torch.squeeze(tensor.logits[:,int(time*len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    dev_feature_list_spoof.append(tensor2)


test_feature_list_bonafide = []
test_feature_list_spoof = []

for i in tqdm(range(len(test_list_bonafide))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_eval/flac/'+test_list_bonafide[i])
    sinc_filtered = sinc(torch.Tensor(pad_random(audio[0], 64600)).unsqueeze(dim=0))
    tensor = model(sinc_filtered.cuda())
    tensor2 = torch.squeeze(tensor.logits[:,int(time*len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    test_feature_list_bonafide.append(tensor2)

for i in tqdm(range(len(test_list_spoof))):
# for i in range(2):
    audio = sf.read('/scratch/r871a03/duneag2/spoof/LA/ASVspoof2019_LA_eval/flac/'+test_list_spoof[i])
    sinc_filtered = sinc(torch.Tensor(pad_random(audio[0], 64600)).unsqueeze(dim=0))
    tensor = model(sinc_filtered.cuda())
    tensor2 = torch.squeeze(tensor.logits[:,int(time*len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    test_feature_list_spoof.append(tensor2)


X_train_bonafide = np.vstack(train_feature_list_bonafide)
X_train_spoof = np.vstack(train_feature_list_spoof)

X_dev_bonafide = np.vstack(dev_feature_list_bonafide)
X_dev_spoof = np.vstack(dev_feature_list_spoof)

X_test_bonafide = np.vstack(test_feature_list_bonafide)
X_test_spoof = np.vstack(test_feature_list_spoof)


train_bonafide_df = pd.DataFrame(data=X_train_bonafide)
train_bonafide_df['target'] = 0

train_spoof_df = pd.DataFrame(data=X_train_spoof)
train_spoof_df['target'] = 1

dev_bonafide_df = pd.DataFrame(data=X_dev_bonafide)
dev_bonafide_df['target'] = 0

dev_spoof_df = pd.DataFrame(data=X_dev_spoof)
dev_spoof_df['target'] = 1

test_bonafide_df = pd.DataFrame(data=X_test_bonafide)
test_bonafide_df['target'] = 0

test_spoof_df = pd.DataFrame(data=X_test_spoof)
test_spoof_df['target'] = 1



train_X_df = pd.concat([train_bonafide_df, train_spoof_df])
train_X_df = train_X_df.sample(frac=1).reset_index(drop=True)
train_y_label = train_X_df["target"]
train_X_df = train_X_df.drop(labels='target', axis=1)

dev_X_df = pd.concat([dev_bonafide_df, dev_spoof_df])
dev_X_df = dev_X_df.sample(frac=1).reset_index(drop=True)
dev_y_label = dev_X_df["target"]
dev_X_df = dev_X_df.drop(labels='target', axis=1)

test_X_df = pd.concat([test_bonafide_df, test_spoof_df])
test_X_df = test_X_df.sample(frac=1).reset_index(drop=True)
test_y_label = test_X_df["target"]
test_X_df = test_X_df.drop(labels='target', axis=1)

# print(train_X_df.shape)
# print(dev_X_df.shape)
# print(test_X_df.shape)

dtrain = xgb.DMatrix(data = train_X_df.values, label = train_y_label)
dval = xgb.DMatrix(data = dev_X_df.values, label = dev_y_label)
dtest = xgb.DMatrix(data = test_X_df.values, label = test_y_label)


params = {
    "max_depth": 7, 
    "eta": 0.1, 
    "objective": "binary:logistic", 
    "eval_metric": "logloss"
    # "early_stoppings": 100 
}

num_rounds = 700

wlist = [(dtrain, "train"), (dval, "val")]

xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = num_rounds,
                      evals = wlist)

pred_probs = xgb_model.predict(dtest)
preds = [1 if x > 0.5 else 0 for x in pred_probs]

pred_probs_val = xgb_model.predict(dval)
preds2 = [1 if x > 0.5 else 0 for x in pred_probs_val]

def get_clf_eval(y_test, pred=None, pred_proba_po=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba_po)

    fpr, tpr, _ = roc_curve(y_test, pred_proba_po)
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    eer = eer * 100
   
    print(confusion)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}, EER (%): {eer:.4f}")

get_clf_eval(test_y_label, pred = preds, pred_proba_po = pred_probs)
get_clf_eval(dev_y_label, pred = preds2, pred_proba_po = pred_probs_val)


#full_save_dir = os.path.join(save_dir, 'iwax_sinc_') + str(low_freq) + '_' + str(high_freq) + '_' + str(time) + '_4_normalized_new.model'

#print('full_save_dir: ', full_save_dir)

#xgb_model.save_model(full_save_dir)
