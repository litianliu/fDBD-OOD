import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
import pdb

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# load collected features

class_num = 10
id_train_size = 50000
id_val_size = 10000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 512))).to(device)
score_log = torch.from_numpy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))).to(device)
label_log = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))).to(device)


cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 512))).to(device)
score_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))).to(device)
label_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))).to(device)


ood_feat_score_log = {}
ood_dataset_size = {
    'SVHN':26032,
    'iSUN': 8925,
    'places365': 10000,
    'dtd': 5640
}

ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    ood_feat_log = torch.from_numpy(np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 512))).to(device)
    ood_score_log = torch.from_numpy(np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], class_num))).to(device)
    ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log 


######## get w, b; precompute demoninator matrix, training feature mean  #################

if args.name == 'resnet18':
    checkpoint = torch.load('ckpt/CIFAR10_resnet18.pth.tar')
    w = checkpoint['state_dict']['fc.weight']
    b = checkpoint['state_dict']['fc.bias']
elif args.name == 'resnet18-supcon':
    checkpoint = torch.load('ckpt/CIFAR10_resnet18_supcon_linear.pth')
    w = checkpoint['model']['fc.weight']
    b = checkpoint['model']['fc.bias']

train_mean = torch.mean(feat_log, dim= 0).to(device)

denominator_matrix = torch.zeros((10,10)).to(device)
for p in range(10):
  w_p = w - w[p,:]
  denominator = torch.norm(w_p, dim=1)
  denominator[p] = 1
  denominator_matrix[p, :] = denominator


#################### fDBD score OOD detection #################

all_results = []
all_score_out = []

values, nn_idx = score_log_val.max(1)
logits_sub = torch.abs(score_log_val - values.repeat(10, 1).T)
#pdb.set_trace()
score_in = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log_val - train_mean , dim = 1)
score_in = score_in.float().cpu().numpy()

for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
    values, nn_idx = score_log.max(1)
    logits_sub = torch.abs(score_log - values.repeat(10, 1).T)
    scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
    scores_out_test = scores_out_test.float().cpu().numpy()
    all_score_out.extend(scores_out_test)
    results = metrics.cal_metric(score_in, scores_out_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, 'fDBD')
print()



# #################### SSD+ score OOD detection #################
# begin = time.time()
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
# prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
# ftrain_ssd = prepos_feat_ssd(ftrain)
# ftest_ssd = prepos_feat_ssd(ftest)
# food_ssd_all = {}
# for ood_dataset in args.out_datasets:
#     food_ssd_all[ood_dataset] = prepos_feat_ssd(food_all[ood_dataset])
#
# inv_sigma_cls = [None for _ in range(class_num)]
# covs_cls = [None for _ in range(class_num)]
# mean_cls = [None for _ in range(class_num)]
# cov = lambda x: np.cov(x.T, bias=True)
# for cls in range(class_num):
#     mean_cls[cls] = ftrain_ssd[label_log == cls].mean(0)
#     feat_cls_center = ftrain_ssd[label_log == cls] - mean_cls[cls]
#     inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
#
# def maha_score(X):
#     score_cls = np.zeros((class_num, len(X)))
#     for cls in range(class_num):
#         inv_sigma = inv_sigma_cls[cls]
#         mean = mean_cls[cls]
#         z = X - mean
#         score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
#     return score_cls.max(0)
#
# dtest = maha_score(ftest_ssd)
# all_results = []
# for name, food in food_ssd_all.items():
#     print(f"SSD+: Evaluating {name}")
#     dood = maha_score(food)
#     results = metrics.cal_metric(dtest, dood)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, 'SSD+')
# print(time.time() - begin)

