from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pretrainedmodels
import time
import glob
import os
from dataset import EvalAtlasData
import argparse
from torch import nn

def construct_rgby_model(model):
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (4, ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(4, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data 
    layer_name = list(container.state_dict().keys())[0][:-7] 

    setattr(container, layer_name, new_conv)
    return model


def generate_preds(model_name,threshold,suffix):
    model = pretrainedmodels.__dict__[model_name](num_classes = 1000, pretrained = 'imagenet')
    in_features = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(in_features, 28)

    if 'resnext' in model_name:
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    if 'resnet' in model_name:
        model.avgpool = nn.AdaptiveAvgPool2d(1)

    model = construct_rgby_model(model)
    
    model.load_state_dict(torch.load('{}_rgby_focal_0.pth.tar'.format(model_name))['state_dict'])
    model = model.eval()
    model.cuda()

    eval_data = EvalAtlasData(model = model_name)
    dataloader = DataLoader(eval_data, 1, False)
    
    preds = []
    for i, (image_id, images) in enumerate(dataloader):
        images = images.cuda()

        raw_predictions = (model(images))
        
        predictions = np.argwhere(raw_predictions.data[0] > threshold)
        try:
            num_predictions = len(predictions.data[0])
        except IndexError:
            num_predictions = 0

        print('-----------------------------------------------------')
        # print(image_id[0])
        # print('Raw Prediction', raw_predictions)
        if num_predictions == 0:
            # print('No value passed the threshold')
            predictions = [np.argmax(raw_predictions.detach().cpu().numpy())]
            num_predictions = 1
            # print("Prediction:", predictions)
            # print("Number of predictions", num_predictions)
        else:
            predictions = predictions.data[0].tolist()
            # print("Prediction:", predictions)
            # print("Number of predictions", num_predictions)

        predicted = ' '.join('%d' % prediction for prediction in predictions)
        print(image_id[0])
        print(predicted)
        pred = dict(Id = image_id[0], Predicted = predicted)
        preds.append(pred)
        
    df = pd.DataFrame(preds)
    df.to_csv('{}_{}_{}.csv'.format(model_name, threshold, suffix), index = False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type = str)
    args = parser.parse_args()

    model_name = args.arch
    threshold = 0
    suffix = 'raw'

    generate_preds(model_name, threshold, suffix)
