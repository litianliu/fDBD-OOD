#! /usr/bin/env python3

import torch

import os

from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True)

batch_size = args.batch_size

FORCE_RUN = True

dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
feat = model.penult_feature(dummy_input)
featdim = feat.shape[1]

#begin = time.time()

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:
    #print('ID ' + split)
    #print(len(in_loader.dataset))

    cache_dir = f"cache/{args.in_dataset}_{split}_{args.name}_in"
    if FORCE_RUN or not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), featdim))
        score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), num_classes))
        label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset),))

        model.eval()

        for batch_idx, (inputs, targets) in enumerate(in_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

            with torch.no_grad():
                out = model.penult_feature(inputs)
                score = model.fc(out)
            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(in_loader)}")

    else:
        feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), featdim))
        score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), num_classes))
        label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset),))

for ood_dataset in args.out_datasets:    

    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader
    #print(ood_dataset)
    #print(len(out_loader.dataset))
    cache_dir = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out"
    if FORCE_RUN or not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), featdim))
        score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), num_classes))

        model.eval()
        for batch_idx, (inputs, _) in enumerate(out_loader):
            inputs = inputs.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

            with torch.no_grad():
                out = model.penult_feature(inputs)
                score = model.fc(out)
            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(out_loader)}")
    else:
        ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), featdim))
        ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), num_classes))

#print(time.time() - begin)