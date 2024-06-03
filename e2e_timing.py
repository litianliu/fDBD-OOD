#! /usr/bin/env python3

import torch
import time
import os
import pdb
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import pdb

print('Preparing....')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True) # set true to load from ash_ckpt in their repo; essentially the same as torch ckpt
model.to(device)

batch_size = args.batch_size

#########################################

id_train_size = 50000
cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 512))).to(device)

checkpoint = torch.load('ckpt/CIFAR10_resnet18.pth.tar')
w = checkpoint['state_dict']['fc.weight']
b = checkpoint['state_dict']['fc.bias']

train_mean = torch.mean(feat_log, dim= 0).to(device)

denominator_matrix = torch.zeros((10,10)).to(device)
for p in range(10):
  w_p = w - w[p,:]
  denominator = torch.norm(w_p, dim=1)
  denominator[p] = 1
  denominator_matrix[p, :] = denominator

model.to(device)
model.eval()
########################################


print("Standalone classification inference begin...")
begin = time.time()
for i in range(10):
    print(f"{i}/10")
    for split, in_loader in [('val', testloaderIn)]:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                score = model(inputs)
                values, nn_idx = score.max(1)

print(f"Standalone Classification Inference: {100*(time.time() - begin)/len(testloaderIn.dataset)} ms per image")

print("Classification Inference + fDBD Score Computation begin...")
begin = time.time()
for i in range(10):
    print(f"{i}/10")
    for split, in_loader in [('val', testloaderIn)]:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                feat, score = model(inputs, return_feature=True)
                values, nn_idx = score.max(1)
                # fDBD score compute
                fDBD_score = torch.sum(torch.abs(score - values.repeat(10, 1).T)/denominator_matrix[nn_idx], axis=1)/torch.norm(feat - train_mean , dim = 1)

print(f"Classification Inference + fDBD Score Computation: {100*(time.time() - begin)/len(testloaderIn.dataset)} ms per image")


