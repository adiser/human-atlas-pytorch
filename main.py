import torch 
import torch.nn as nn
import pretrainedmodels 
import numpy as np 
import time 
import glob 
import argparse
import models
from dataset import AtlasData
from metrics import *
from torch.utils.data import Dataset, DataLoader


def train_and_val(model_name, split, batch_size, epochs, lr, start_epoch):

    model = models.construct_rgby_model(model_name, split)
    model.cuda()

    train_dataset = AtlasData(split = split, train = True, model = model_name)
    val_dataset = AtlasData(split = split, train = False, model = model_name)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    
    f2_loss = SmoothF2Loss()
    focal_loss = FocalLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(start_epoch,epochs+1):
        
        train(model, train_loader, optimizer, f2_loss, focal_loss, epoch)
        avg_loss, f1_score = validate(model, val_loader, f2_loss, focal_loss, epoch)
                    
        if epoch % 10 == 0:
            filename = '{}_rgby_focal_{}_{}.pth.tar'.format(model_name, split, epoch)
        else:
            filename = '{}_rgby_focal_{}.pth.tar'.format(model_name, split)
            
        state = {'loss': avg_loss, 'f1_score': f1_score, 'epoch': epoch+1, 'state_dict': model.state_dict()}           
        torch.save(state, filename)

    
def train(model, train_loader, optimizer, f2_loss, focal_loss, epoch):
    
    model.train()
    start = time.time()
    losses = []
    for i, (images, label_arrs, labels) in enumerate(train_loader):
        images = images.cuda()
        label_arrs = label_arrs.cuda()

        outputs = model(images)

        loss = f2_loss(outputs, label_arrs)
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
            

def validate(model, val_loader, f2_loss, focal_loss, epoch):
    
    model.eval()
    
    losses = []
    y_pred = np.zeros(len(val_loader) * 28).reshape(len(val_loader), 28)
    y_true = np.zeros(len(val_loader) * 28).reshape(len(val_loader), 28)

    for i, (images, label_arrs, labels) in enumerate(val_loader):
        images = images.cuda()
        label_arrs_cuda = label_arrs.cuda()

        raw_predictions = model(images)
        outputs = raw_predictions.data
        
        loss = f2_loss(outputs, label_arrs_cuda) + focal_loss(outputs, label_arrs_cuda)
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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type = str, choices = models.model_configs.keys())
    args = parser.parse_args()
    
    model_name = args.arch
    batch_size = 16
    epochs = 100
    start_epoch = 1
    lr = 0.0001

    for split in range(3):
        train_and_val(model_name, split, batch_size, epochs, lr, start_epoch)

if __name__ == "__main__":
    main()
        