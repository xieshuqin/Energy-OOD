# Energy-OOD
This repo is a PyTorch implementation of the paper [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759)

## Installation
- Download the Tiny-images dataset and put them under `data` folder
- Setup environment.
 
    Assuming PyTorch is already installed. 
    
    `pip install tensorboard ood-metrics`

## Instructions
- Run `python main.py` to train a model using CIFAR10 as in-distribution dataset 
and SVHN as out-of-distribution dataset. 