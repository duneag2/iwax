import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import os
import pickle
import sys
import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import optuna
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from collections import OrderedDict
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import json
import argparse
from importlib import import_module
import random
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description="Select model type")
parser.add_argument("--save_dir", type=str, required=True, help="where to save the model file")
parser.add_argument("--w2v2", type=str, required=True, help="w2v2 weight path")

args = parser.parse_args()
save_dir = args.save_dir
path_to_w2v2 = args.w2v2


# path to aasist (w2v2 class)
path_to_w2v2_class = './aasist'
sys.path.insert (0, path_to_w2v2_class)
sys.path.insert(0, path_to_w2v2)


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


with open('./aasist/config/wav2vec2_aasist_base.conf', "r") as f_json:
    config = json.loads(f_json.read())

model_config = config["model_config"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w2v2 = get_model(model_config, device)

state_dict = torch.load(path_to_w2v2, map_location=device)

filtered_state_dict = {k: v for k, v in state_dict.items() if k != "pos_S"}



w2v2.load_state_dict(filtered_state_dict)

model = torch.nn.Sequential(*list(w2v2.children())[:-17])


#device = "cuda" if torch.cuda.is_available() else "cpu"
#w2v2 = torch.load(path_to_w2v2, map_location=device)

#model = torch.nn.Sequential(OrderedDict(w2v2))
#model = torch.nn.Sequential(*list(model.children())[:-17])

# model = torch.nn.Sequential(*list(w2v2.children())[:-17])


train_list_bonafide = []
train_list_spoof = []
with open("./LA2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "r") as f:
    for line in f:
      if line.split()[-1] == 'bonafide':
        train_list_bonafide.append(line.split()[1]+'.flac')
      else:
        train_list_spoof.append(line.split()[1]+'.flac')


dev_list_bonafide = []
dev_list_spoof = []
with open("./LA2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "r") as f:
    for line in f:
      if line.split()[-1] == 'bonafide':
        dev_list_bonafide.append(line.split()[1]+'.flac')
      else:
        dev_list_spoof.append(line.split()[1]+'.flac')


test_list_bonafide = []
test_list_spoof = []
with open("./LA2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "r") as f:
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

train_feature_list_bonafide = []
train_feature_list_spoof = []

for i in tqdm(range(len(train_list_bonafide))):
# for i in range(2): # for integrity check
    audio = sf.read('./LA/ASVspoof2019_LA_train/flac/'+train_list_bonafide[i])
    tensor = model(torch.unsqueeze(torch.Tensor(pad_random(audio[0], 64600)).cuda(),dim=0))
    for j in range(len(tensor)):
        tensor2 = torch.squeeze(tensor.logits[:,j,:],dim=0)
        tensor2 = tensor2.detach().cpu().numpy()
        train_feature_list_bonafide.append(tensor2)

for i in tqdm(range(len(train_list_spoof))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_train/flac/'+train_list_spoof[i])
    tensor = model(torch.unsqueeze(torch.Tensor(pad_random(audio[0], 64600)).cuda(),dim=0))
    for j in range(len(tensor)):
        tensor2 = torch.squeeze(tensor.logits[:,j,:],dim=0)
        tensor2 = tensor2.detach().cpu().numpy()
        train_feature_list_spoof.append(tensor2)


dev_feature_list_bonafide = []
dev_feature_list_spoof = []

for i in tqdm(range(len(dev_list_bonafide))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_dev/flac/'+dev_list_bonafide[i])
    tensor = model(torch.unsqueeze(torch.Tensor(pad_random(audio[0], 64600)).cuda(),dim=0))
    tensor2 = torch.squeeze(tensor.logits[:,int(len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    dev_feature_list_bonafide.append(tensor2)

for i in tqdm(range(len(dev_list_spoof))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_dev/flac/'+dev_list_spoof[i])
    tensor = model(torch.unsqueeze(torch.Tensor(pad_random(audio[0], 64600)).cuda(),dim=0))
    tensor2 = torch.squeeze(tensor.logits[:,int(len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    dev_feature_list_spoof.append(tensor2)


test_feature_list_bonafide = []
test_feature_list_spoof = []

for i in tqdm(range(len(test_list_bonafide))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_eval/flac/'+test_list_bonafide[i])
    tensor = model(torch.unsqueeze(torch.Tensor(pad_random(audio[0], 64600)).cuda(),dim=0))
    tensor2 = torch.squeeze(tensor.logits[:,int(len(tensor)/4),:],dim=0)
    tensor2 = tensor2.detach().cpu().numpy()
    test_feature_list_bonafide.append(tensor2)

for i in tqdm(range(len(test_list_spoof))):
# for i in range(2):
    audio = sf.read('./LA/ASVspoof2019_LA_eval/flac/'+test_list_spoof[i])
    tensor = model(torch.unsqueeze(torch.Tensor(pad_random(audio[0], 64600)).cuda(),dim=0))
    tensor2 = torch.squeeze(tensor.logits[:,int(len(tensor)/4),:],dim=0)
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

def get_clf_eval(y_test, pred=None, pred_proba_po=None):
    # confusion = confusion_matrix(y_test, pred)
    # accuracy = accuracy_score(y_test, pred)
    # precision = precision_score(y_test, pred)
    # recall = recall_score(y_test, pred)
    # f1 = f1_score(y_test, pred)
    # auc = roc_auc_score(y_test, pred_proba_po)

    fpr, tpr, _ = roc_curve(y_test, pred_proba_po)
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    eer = eer * 100
   
    # print(confusion)
    print(f"EER (%): {eer:.4f}")

get_clf_eval(test_y_label, pred = preds, pred_proba_po = pred_probs)

xgb_model.save_model(save_dir)
