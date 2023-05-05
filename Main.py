import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import models
import Properties as p
from Net import Net
import os
from tqdm import tqdm
from ImageDataset import ImageDataset
from torchvision.datasets import ImageFolder
from Evaluation import Evaluation
import click
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class Main:
    def __init__(self, batch_size, num_workers, epochs, lr, wd, gamma, save_model, im_size):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.gamma = gamma
        self.save_model = save_model
        self.im_size = im_size

        train_dir = "Data/train/"
        val_dir = "Data/val/"
        train_dataset = ImageDataset(train_dir)
        val_dataset = ImageDataset(val_dir)

        model = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.train(model, train_dataset, val_dataset, criterion, optimizer)

    def train(self, model, train_data, val_data, criterion, optimizer):

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=self.num_workers, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                                     pin_memory=True)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model.to(device)
            model.train()

        for epoch_num in range(self.epochs):
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
                f'\nEpochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_dataloader) * self.batch_size): .3f} \
                    | Train Accuracy: {total_acc_train / (len(train_dataloader) * self.batch_size): .3f} \
                    | Val Loss: {total_loss_val / (len(val_dataloader) * self.batch_size): .3f} \
                    | Val Accuracy: {total_acc_val / (len(val_dataloader) * self.batch_size): .3f}')

        if self.save_model:
            if not os.path.exists("Results"):
                os.mkdir("Results")
            if not os.path.exists("Results/{}".format("custom")):
                os.mkdir("Results/{}".format("custom"))
            torch.save(model, "Results/{}/{}_model.pt".format("custom", "custom"))


@click.command()
@click.option('--batch_size', default=4, help='Batch Size')
@click.option('--num_workers', default=4, help='Num Workers')
@click.option('--epochs', default=9, help='Epochs')
@click.option('--lr', default=0.001, help='Learning Rate')
@click.option('--wd', default=0, help='Weight Decay')
@click.option('--gamma', default=0.9, help='Gamma')
@click.option('--save', '-s', is_flag=True, help="Save Model at the end")
@click.option('--im_size', default=250, help='Image Size')
def main(batch_size, num_workers, epochs, lr, wd, gamma, save, im_size):
    Main(batch_size, num_workers, epochs, lr, wd, gamma, save, im_size)


if __name__ == "__main__":
    main()
