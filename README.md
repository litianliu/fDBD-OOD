# Fast Decision Boundary based Out-of-Distribution Detector

This repository contains code for the paper [Fast Decision Boundary based Out-of-Distribution Detector](https://arxiv.org/abs/2312.11536) (ICML 2024) by Litian Liu and Yao Qin.

Besides this codebase, we also demo the implementation and performance of our algorithm fDBD on [OpenOOD Benchmark](https://github.com/Jingkang50/OpenOOD/tree/main). See [Colab Version](https://colab.research.google.com/drive/1ebGFVrLZJ2HpO5R-VTkhG2akK3oqsOxK?usp=sharing) here.  

<div style="text-align:center;">
    <img src="img/intro_pic.pdf" alt="intro" width="50%" height="50%">
</div>

## Setup

```bash
# create conda env and install dependencies
$ conda env create -f environment.yml
$ conda activate fdbd
# set environmental variables
$ export DATASETS='data'
# download datasets and checkpoints
$ bash download.sh
```
Please download ImageNet dataset manually to your own `$IMAGENET` dir by following [this](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) instructions.

## Demo

CIFAR-10 Benchmark
```
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN iSUN dtd places365 --name resnet18-supcon  --model-arch resnet18-supcon

python run_cifar.py --in-dataset CIFAR-10  --out-datasets SVHN iSUN dtd places365 --name resnet18-supcon   --model-arch resnet18-supcon 
```

ImageNet Benchmark
```
python feat_extract_largescale.py --in-dataset imagenet --imagenet-root $IMAGENET --out-datasets inat sun50 places50 dtd  --name resnet50-supcon --model-arch resnet50-supcon

python run_imagenet.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name resnet50-supcon  --model-arch resnet50-supcon
```

fDBD with activation shaping
```
python feat_extract_largescale.py --in-dataset imagenet --imagenet-root /data/chenhe/datasets/imagenet --out-datasets inat sun50 places50 dtd  --name resnet50 --model-arch resnet50

python run_imagenet_w_ASH.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50
```

Inference Efficiency
```
python e2e_timing.py --in-dataset CIFAR-10 --name resnet18 --model-arch resnet18
```

## References

```bibtex
@article{djurisic2022ash,
    url = {https://arxiv.org/abs/2209.09858},
    author = {Djurisic, Andrija and Bozanic, Nebojsa and Ashok, Arjun and Liu, Rosanne},
    title = {Extremely Simple Activation Shaping for Out-of-Distribution Detection},
    publisher = {arXiv},
    year = {2022},
    }
```

```bibtex
@article{sun2022knnood,
  title={Out-of-distribution Detection with Deep Nearest Neighbors},
  author={Sun, Yiyou and Ming, Yifei and Zhu, Xiaojin and Li, Yixuan},
  journal={ICML},
  year={2022}
}
```
      
## Citations

Please cite our paper if you find this codebase helpful! 

```bibtex
@article{liu2024fast,
  title={Fast Decision Boundary based Out-of-Distribution Detector},
  author={Liu, Litian and Qin, Yao},
  journal={ICML},
  year={2024}
}
```
