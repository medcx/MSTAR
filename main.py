import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from model import ResNet18
from data.dataload import create_dataloaders
from utils import set_seed, get_parser
from process.train import train_process
from process.test import test_process



if __name__ == '__main__':
    args = get_parser()
    set_seed(args.seed)

    # create dataloader
    dataloaders, dataset_sizes = create_dataloaders(args.data_path)
    if args.use_gpu:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Training and validation device : {device}')

    # init the network
    ResNet = ResNet18()
    model = ResNet.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)


    if args.train:
        # train process
        train_process(args, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler)
    else:
        # test process
        test_process(args, model, dataloaders, dataset_sizes)