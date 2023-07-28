import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import os
import json
import pathlib
import math

from dataset import IPN, MyDataset
from model.TD_Net import TDNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-split', type=str, default='train')
parser.add_argument('-dataset', type=str, default='ipn')
parser.add_argument('-root', type=str, default='no_root')
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-n_joints', type=int, default=42)
parser.add_argument('-n_frames', type=int, default=60)
parser.add_argument('-n_filters', type=int, default=64)
parser.add_argument('-feat_dims', type=int, default=210)
parser.add_argument('-num_workers', type=int, default=0)
parser.add_argument('-is_balanced', type=str, default='False')
parser.add_argument('-transform', type=str, default='False')
parser.add_argument('-weighted', type=str, default='False')
args = parser.parse_args()


# set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)

torch.set_default_dtype(torch.float64)


def main():
    args.is_balanced = eval(args.is_balanced)
    args.weighted = eval(args.weighted)
    args.transform = eval(args.transform)
    
    print(args)

    n_frames = args.n_frames
    n_joints = args.n_joints
    joint_dims = 3
    n_filters = args.n_filters
    feat_dims = args.feat_dims
    n_classes = 20 
    drop_rate = 0.75

    params = [n_joints,
              joint_dims,
              feat_dims,
              n_filters,
              n_frames,
              n_classes,
              drop_rate]

    model = TDNet(*params)


    datasets = get_datasets(args.dataset, args.n_frames, n_joints, n_classes, args.root, args.is_balanced, args.transform)
    if args.batch_size == 0:
        args.batch_size = int(len(datasets['train']) / 6)
    dataloaders = get_dataloaders(args.batch_size, datasets, args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if args.weighted:
        weight = torch.ones(20).to(device)
        weight[0] = 1/20
        criterion = nn.NLLLoss(weight=weight)
        print('Training with weighted NLL loss')
    else:
        criterion = nn.NLLLoss()
    if args.is_balanced:
        print('Balanced Dataset with Reduced class 0')
    if not args.weighted and not args.is_balanced:
        print('Default settings')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7, min_lr=1e-5, verbose=True)

    model.to(device)

    if args.load_model:
        print(f'Loading from {args.load_model} ...')
        checkpoint = load_weight(args.load_model)

        model.load_state_dict(checkpoint['model'])

        start_epoch = checkpoint['epoch']
        lr_sched.load_state_dict(checkpoint['lr_sched'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print(f'Start from epoch {start_epoch}')
    else:
        start_epoch = 1

    print('\n')
    training_loop(args.epoch, model, dataloaders, criterion, optimizer, lr_sched, device, start_epoch)



def training_loop(num_epoch, model, dataloaders, criterion, optimizer, lr_sched, device, start_epoch):
    since = time.perf_counter()
    history = {}
    best_val_loss = float('inf')
    best_val_acc = 0
    
    root = 'result/checkpoint'

    name_str = ''
    if args.weighted:
        name_str += 'w'
    if args.is_balanced:
        name_str += 'b'
    if not args.weighted and not args.is_balanced:
        name_str += 'd'
    name_str = f'{name_str}_{args.n_frames}_{args.n_filters}_{args.n_joints}'
    print(f'Model name: {name_str}')

    checkpoint_path = os.path.join(root, name_str)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, num_epoch + 1):
        since1 = time.perf_counter()
        print(f'Epoch {epoch}/{num_epoch}')

        train_loss, train_acc = train_step(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc, preds, targets = val_step(model, dataloaders['val'], criterion, device)
        lr_sched.step(val_loss)

        # Create logging files
        history[epoch] = {'Train loss': train_loss, 'Val loss': val_loss, 
                          'Train accuracy': train_acc, 'Val accuracy': val_acc}
        log_result(history, name_str, 'loss' )

        # Save model weights every 10 epochs
        if epoch % 20 == 0:
            model_name = f'_{epoch}.pth'
            dump_weight(os.path.join(checkpoint_path, model_name), model, optimizer, lr_sched, epoch+1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best {val_loss=} at epoch {epoch}')
            model_name = f'loss_{epoch}.pth'
            dump_weight(os.path.join(checkpoint_path, model_name), model, optimizer, lr_sched, epoch+1)

        if val_acc > best_val_acc:
            print(f'Best {val_acc=} % at epoch {epoch}')

            model_name = f'acc_{epoch}.pth'
            dump_weight(os.path.join(checkpoint_path, model_name), model, optimizer, lr_sched, epoch+1)

            create_offline_logging(targets, preds, name_str)
            best_val_acc = val_acc
            

        print("Epoch", epoch,
              "Total_Time", round(time.perf_counter() - since),
              "Epoch_time", round(time.perf_counter() - since1))
        print()
    print(f'Best acc: {best_val_acc}')
    print(f'Best loss: {best_val_loss}')


def forward(model, pose, motion, target, criterion):
    pose = pose.float()
    motion = motion.float()
    output = model(pose, motion)
    loss = criterion(output, target)
    return loss


def train_step(model, dataloader, criterion, optimizer, device):
    tot_loss = 0
    correct = n_instances = 0
    
    for counter, ([pose, motion], target) in enumerate(dataloader, start=1):
        pose, motion, target = pose.to(device), motion.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(pose, motion)
        output = F.log_softmax(output, dim=1)
        loss = criterion(output, target)

        tot_loss += loss.detach()
        correct += (torch.argmax(output, dim=1) == target).detach().sum()
        n_instances += target.size(0)

        loss.backward()
        optimizer.step()
        
    epoch_loss = (tot_loss / counter).item()
    acc = correct / n_instances
    acc = round((acc*100).item(), 4)
    print('Train loss: ', epoch_loss)
    print(f'Train accuracy: {acc}%')
    
    return epoch_loss, acc


def val_step(model, dataloader, criterion, device):
    tot_loss = 0
    correct = 0
    preds = []
    targets = []

    with torch.no_grad():
        model.eval()

        for counter, ([pose, motion], target) in enumerate(dataloader, start=1):
            pose, motion, target = pose.to(device), motion.to(device), target.to(device)
            output = model(pose, motion)

            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)

            tot_loss += loss.item()
            correct += (torch.argmax(output, dim=1) == target).float().sum()
        
            preds.append(torch.argmax(output, dim=1).item())
            targets.append(target.item())
	
        print(f'\tPredict: {preds[:20]}')
        print(f'\tTargets: {targets[:20]}')
	
        epoch_loss = tot_loss / counter
        acc = correct / len(dataloader)
        acc = round((acc*100).item(), 4)

    print('Val loss: ', epoch_loss)
    print(f'Val accuracy: {acc}%')

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    return epoch_loss, acc, preds, targets


if __name__ == '__main__':
    main()
