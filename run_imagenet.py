import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
import torchvision.models as models

args = get_args()

seed = args.seed
print(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class_num = 1000
id_train_size = 1281167
id_val_size = 50000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 2048))).to(device)
score_log = torch.from_numpy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))).to(device)
label_log = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))).to(device)


cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 2048))).to(device)
score_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))).to(device)
label_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))).to(device)


ood_feat_score_log = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}

for ood_dataset in args.out_datasets:
    ood_feat_log = torch.from_numpy(np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 2048))).to(device)
    ood_score_log = torch.from_numpy(np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], class_num))).to(device)
    ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log 


######## get w, b; precompute demoninator matrix, training feature mean  #################

if args.name == 'resnet50':
    net = models.resnet50(pretrained=True)
    for i, param in enumerate(net.fc.parameters()):
      if i == 0:
        w = param.data
      else:
        b = param.data

elif args.name == 'resnet50-supcon':
    checkpoint = torch.load('ckpt/ImageNet_resnet50_supcon_linear.pth')
    w = checkpoint['model']['fc.weight']
    b = checkpoint['model']['fc.bias']

train_mean = torch.mean(feat_log, dim= 0).to(device)

denominator_matrix = torch.zeros((1000,1000)).to(device)
for p in range(1000):
  w_p = w - w[p,:]
  denominator = torch.norm(w_p, dim=1)
  denominator[p] = 1
  denominator_matrix[p, :] = denominator

#################### fDBD score OOD detection #################

all_results = []
all_score_out = []

values, nn_idx = score_log_val.max(1)
logits_sub = torch.abs(score_log_val - values.repeat(1000, 1).T)
#pdb.set_trace()
score_in = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log_val - train_mean , dim = 1)
score_in = score_in.float().cpu().numpy()

for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
    values, nn_idx = score_log.max(1)
    logits_sub = torch.abs(score_log - values.repeat(1000, 1).T)
    scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
    scores_out_test = scores_out_test.float().cpu().numpy()
    all_score_out.extend(scores_out_test)
    results = metrics.cal_metric(score_in, scores_out_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, 'fDBD')
print()