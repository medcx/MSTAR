import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from model import ResNet18
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import random
import seaborn as sns



def test_process(args, model, dataloaders, dataset_sizes):
    model.load_state_dict(torch.load(args.checkpoint))
    criterion = nn.CrossEntropyLoss()

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0.0
    all_preds = []
    all_labels = []

    # Iterate over data.
    t = tqdm(dataloaders['test'])
    for data in t:
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        if args.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects / dataset_sizes['test']
    cm = confusion_matrix(all_labels, all_preds)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))
    print('val Acc: {:4f}'.format(epoch_acc))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{args.save_path}/confusion_matrix.png')
