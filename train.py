import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import models

import CFG as cfg
from Net import Net
from Utils_Model import Utils_Model
import os
import numpy as np
from tqdm import tqdm
from ImageDataset import ImageDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from AccSGD import *
from MomentumOptimizer import *


def train():

    model = Net()
    # model = getattr(models, "resnet18")()
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, p.NUM_CLASSES)
    # model = models.densenet121()
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, cfg.NUM_CLASSES)
    # torch.nn.init.kaiming_normal_(model.classifier.weight)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)

    train_dataset = ImageDataset(cfg.TRAIN_PATH)
    val_dataset = ImageDataset(cfg.VAL_PATH)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.num_workers, pin_memory=True)

    if use_cuda:
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        model.to(device)
        model.train()

    for epoch_num in range(opt.epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_input, train_label = train_input.to(device), train_label.to(device)

            output = model(train_input)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_input, val_label = val_input.to(device), val_label.to(device)

                output = model(val_input)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'\nEpochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_dataloader) * opt.batch_size): .3f} \
                | Train Accuracy: {total_acc_train / (len(train_dataloader) * opt.batch_size): .3f} \
                | Val Loss: {total_loss_val / (len(val_dataloader) * opt.batch_size): .3f} \
                | Val Accuracy: {total_acc_val / (len(val_dataloader) * opt.batch_size): .3f}')
        if epoch_num % 10 == 0:
            Utils_Model(model, cfg.NAME, "epoch_{}".format(epoch_num))

        with open('Results/{}/accloss.txt'.format(cfg.NAME), 'a') as f:
            f.write(f'{epoch_num + 1},{total_acc_train / (len(train_dataloader) * opt.batch_size):.3f}, \
                     {total_loss_train / (len(train_dataloader) * opt.batch_size):.3f}, \
                     {total_acc_val / (len(val_dataloader) * opt.batch_size):.3f}, \
                     {total_loss_val / (len(val_dataloader) * opt.batch_size):.3f}\n')

    Utils_Model(model, cfg.NAME, "epoch_last")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS, help='Num Workers')
    parser.add_argument('--epochs', type=int, default=cfg.EPOCH, help='Epochs')
    parser.add_argument('--lr', type=float, default=cfg.LR, help='Learning Rate')
    parser.add_argument('--wd', type=float, default=cfg.WD, help='Weight Decay')
    parser.add_argument('--gamma', type=float, default=cfg.GAMMA, help='Gamma')
    parser.add_argument('--im_size', type=int, default=cfg.SIZE, help='Image Size')
    opt = parser.parse_args()
    train()
