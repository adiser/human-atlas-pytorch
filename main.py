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


def f1_micro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true
    
    p = truepos.sum() / (preds_bin.sum() + eps) # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps) # take sums and calculate recall on scalars
    
    f1 = 2*p*r / (p+r+eps) # we calculate f1 on scalars
    return f1

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps) # sum along axis=0 (classes)
                                                            # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)    # sum along axis=0 (classes) 
                                                            #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = torch.nn.functional.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

def train_and_val(model_name, split, batch_size, epochs, lr, start_epoch):
    
    model = pretrainedmodels.__dict__[model_name](num_classes = 1000)
    model = construct_rgby_model(model)
            
    num_features = model.last_linear.in_features 
    model.last_linear = torch.nn.Linear(num_features, 28)
    
    if glob.glob('{}_rgby_focal_{}*'.format(model_name, split)):
        pth_file = torch.load('{}_rgby_focal_0.pth.tar'.format(model_name))
        state_dict = pth_file['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = pth_file['epoch']
        
    model.cuda()

    train_dataset = AtlasData(split = split, train = True, model = model_name)
    val_dataset = AtlasData(split = split, train = False, model = model_name)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    
    log_loss = torch.nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(start_epoch,epochs+1):
        
        train(model, train_loader, optimizer, log_loss, focal_loss, epoch)
        avg_loss, f1_score = validate(model, val_loader, log_loss, focal_loss, epoch)
                    
        if epoch % 10 == 0:
            filename = '{}_rgby_focal_{}_{}.pth.tar'.format(model_name, split, epoch)
        else:
            filename = '{}_rgby_focal_{}.pth.tar'.format(model_name, split)
            
        state = {'loss': avg_loss, 'f1_score': f1_score, 'epoch': epoch+1, 'state_dict': model.state_dict()}           
        torch.save(state, filename)

    
def train(model, train_loader, optimizer, log_loss, focal_loss, epoch):
    
    model.train()
    start = time.time()
    losses = []
    for i, (images, label_arrs, labels) in enumerate(train_loader):
        images = images.cuda()
        label_arrs = label_arrs.cuda()

        outputs = model(images)

        loss = focal_loss(outputs, label_arrs)
        losses.append(loss.data[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        elapsed = end-start
        
        if i%100==0:
            print("Epoch [{}], Iteration [{}/{}], Loss: {:.4f} ({:.4f}), Elapsed Time {:.4f}"
                .format(epoch, i+1, len(train_loader), loss.data[0], sum(losses)/len(losses), elapsed))
            
    print("Average Loss: {}".format(sum(losses)/len(losses)))
            

def validate(model, val_loader, log_loss, focal_loss, epoch):
    
    model.eval()
    
    losses = []
    y_pred = np.zeros(len(val_loader) * 28).reshape(len(val_loader), 28)
    y_true = np.zeros(len(val_loader) * 28).reshape(len(val_loader), 28)

    for i, (images, label_arrs, labels) in enumerate(val_loader):
        images = images.cuda()
        label_arrs_cuda = label_arrs.cuda()

        raw_predictions = model(images)
        outputs = raw_predictions.data
        
        loss = focal_loss(outputs, label_arrs_cuda) 
        losses.append(loss.data)
        
        predictions = np.arange(28)[raw_predictions.data[0] > 0.15]
        
        y_pred[i,:] = predictions
        y_true[i,:] = label_arrs
        
        if sum(predictions) == 0:
            prediction = np.argmax(raw_predictions.detach().cpu().numpy())
            predictions = np.zeros(28)
            np.put(predictions, prediction, 1)
        
        
        if i%1000==0:
            print('Testing {}/{}: Loss {}'.format(i, 
                                                 len(val_loader), 
                                                 sum(losses)/len(losses)))
                                                                         
    score = f1_macro(y_true, y_pred)
    avg_loss = sum(losses)/len(losses)
    print("Avg Loss {}".format(avg_loss))
    print("Score {}".format(score))

    return avg_loss, score
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type = str)
    parser.add_argument('--split', type = int)
    args = parser.parse_args()

    model_name = args.arch
    split = args.split
    batch_size = 16
    epochs = 100
    start_epoch = 1
    lr = 0.0001

    train_and_val(model_name, split, batch_size, epochs, lr, start_epoch)

        