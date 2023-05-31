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
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


def test():
    test_dataset = ImageDataset(cfg.TEST_PATH)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True)

    model = torch.load(opt.weights)
    model.eval()
    true_labels = []
    predicted_labels = []

    for inputs, labels in test_dataloader:
        with torch.no_grad():
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    get_metrics(true_labels, predicted_labels)


def get_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    confusion = confusion_matrix(true_labels, predicted_labels)

    predicted_labels = [int(prob >= 0.5) for prob in predicted_labels]

    n_classes = 3
    true_labels_bin = label_binarize(true_labels, classes=range(n_classes))
    predicted_labels_bin = label_binarize(predicted_labels, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predicted_labels_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    classDict = ({0: "Car", 1: "Truck", 2: "Bus"})
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {classDict.get(i)} (AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
    plt.legend()
    plt.savefig('Results/{}/auc_curve.svg'.format(opt.weights.split("/")[1]))
    export_results(accuracy, precision, recall, f1, confusion)


def export_results(accuracy, precision, recall, f1, confusion):
    if opt.nosave is False:
        print(accuracy, " ", precision, " ", recall, " ", f1, " ", confusion, " ")
        with open('Results/{}/results.txt'.format(opt.weights.split("/")[1]), "w") as f:
            f.write(f"Accuracy: {round(accuracy, 2)}\n")
            f.write(f"Precision: {round(precision, 2)}\n")
            f.write(f"Recall: {round(recall, 2)}\n")
            f.write(f"F1 score: {round(f1, 2)}\n")
            f.write(f"Confusion matrix:\n{confusion}\n")

        # plt.savefig('Results/{}/auc_curve.png'.format(opt.weights.split("/")[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=cfg.WEIGHTS, help='Weights')
    parser.add_argument('--nosave', action='store_true', help='Save test results to a file')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS, help='Num Workers')
    opt = parser.parse_args()
    test()
