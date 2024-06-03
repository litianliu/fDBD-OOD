import os
import pdb
# import models.densenet as dn
# import models.wideresnet as wn


import torch

def get_model(args, num_classes, load_ckpt=True, load_epoch=None):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50-supcon':
            from models.resnet_supcon import SupConResNet
            model = SupConResNet(num_classes=num_classes)
            checkpoint = torch.load('ckpt/ImageNet_resnet50_supcon.pth')
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
            checkpoint_linear = torch.load('ckpt/ImageNet_resnet50_supcon_linear.pth')
            state_dict['fc.weight'] = checkpoint_linear['model']['fc.weight'] 
            state_dict['fc.bias'] = checkpoint_linear['model']['fc.bias'] 
            model.load_state_dict(state_dict)
    else:
        # create model
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method, p=args.p)
            checkpoint = torch.load('ckpt/CIFAR10_resnet18.pth.tar')
            checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}}
            model.load_state_dict(checkpoint['state_dict'])
        elif args.model_arch == 'resnet18-supcon':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
            checkpoint = torch.load('ckpt/CIFAR10_resnet18_supcon.pth.tar')
            checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}}
            checkpoint_linear = torch.load('ckpt/CIFAR10_resnet18_supcon_linear.pth')
            checkpoint['state_dict']['fc.weight'] = checkpoint_linear['model']['fc.weight'] 
            checkpoint['state_dict']['fc.bias'] = checkpoint_linear['model']['fc.bias'] 
            model.load_state_dict(checkpoint['state_dict'])
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        #if load_ckpt:
        #    epoch = args.epochs
        #    if load_epoch is not None:
        #        epoch = load_epoch
            # checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch))
            # checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch), map_location='cpu')
            #checkpoint = torch.load("/home/litianl/Neural-Collapse/model_weights/resnet_adam_sota_cifar10/epoch_{epochs}.pth".format(epochs=epoch), map_location='cpu')
            #pdb.set_trace()
            #checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}}
            #model.load_state_dict(checkpoint['state_dict'])
        #    model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
