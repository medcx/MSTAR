import argparse
import os
import torch
import numpy as np
import random

import matplotlib.pyplot as plt



def set_seed(seed):
    # set random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/", help="folder to load data")
    parser.add_argument("--epochs", type=int, default=300, help="total number of epochs")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--use_gpu", type=bool, default=True, help="training on gpu or not")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--model_save_path", type=str, default='./checkpoint', help="folder to save results")
    parser.add_argument("--save_path", type=str, default='./result', help="folder to save results")
    parser.add_argument("--train", action='store_true', help="train or test process")
    parser.add_argument("--checkpoint", type=str, default='./checkpoint/model_best.pth', help="the path of model weight ")

    args = parser.parse_args()
    return args



def draw(args, train_losses, test_losses, train_accuracies, test_accuracies):
    os.makedirs(args.save_path, exist_ok=True)
    epochs = range(1, args.epochs + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(f'{args.save_path}/train_loss.png')
    plt.close()

    plt.plot(epochs, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(f'{args.save_path}/test_loss.png')
    plt.close()

    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.savefig(f'{args.save_path}/train_acc.png')
    plt.close()

    plt.plot(epochs, test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.savefig(f'{args.save_path}/test_acc.png')
    plt.close()

