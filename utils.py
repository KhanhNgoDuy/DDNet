import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from dataset import IPN, MyDataset, OldDataset


torch.set_default_dtype(torch.float64)


def log_result(history, name, mode):
    root = 'result/logging'
    log_path = os.path.join(root, name)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    if mode == 'loss':
        file_name = 'log_loss.json'
        indent = 4
    elif mode == 'prob':
        file_name = 'log_prob.json'
        indent = None

    path = os.path.join(log_path, file_name)

    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(history, f, indent=indent)
        return

    with open(path, 'r') as f:
        content = json.load(f)
        if len(content) > len(history):
            return
        
    with open(path, 'w') as f:
        json.dump(history, f, indent=indent)


def create_offline_logging(gt, pred, model_name):
    save_dir = Path('result/logging') / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(gt, pred, zero_division=0.0)
    with open(os.path.join(save_dir, 'report.txt'), 'w') as f:
        f.write(report)
    
    cm = confusion_matrix(gt, pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f')
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig((save_dir / 'cm.jpg').as_posix(), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def get_CG(self, p):
    #print(p.shape)
    M = []
    iu = np.triu_indices(42, 1, 42)
    for f in range(self.n_frames):
        #distance max
        d_m = cdist(p[f],np.concatenate([p[f],np.zeros([1, 3])]),'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M


def get_datasets(dataset, n_frames, n_joints, n_classes, root, is_balanced):
    if dataset == 'IPN':
        dataset = IPN
        root = os.path.join(root, 'IPN')
    elif dataset == 'Custom':
        dataset = MyDataset
        root = os.path.join(root, 'MyDataset')
    elif dataset == 'Old':
        dataset = OldDataset
        root = os.path.join(root, 'MyDataset')
    else:
        raise AssertionError('Dataset must be in ["IPN", "Custom"]')

    train_dataset = dataset(
        split='train',
        root=root,
        n_frames=n_frames,
        n_classes=n_classes,
        n_joints=n_joints,
        is_balanced=is_balanced
    )
    val_dataset = dataset(
        split='val',
        root=root,
        n_frames=n_frames,
        n_joints=n_joints,
        n_classes=n_classes,
        is_balanced=False
    )
    datasets = {'train': train_dataset, 'val': val_dataset}
    return datasets


def get_dataloaders(batch_size, datasets, num_workers):
    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    dataloaders = {'train': train_loader, 'val': val_loader}

    return dataloaders


def dump_weight(path, model, optimizer, lr_sched, epoch):

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched.state_dict()
    }

    torch.save(checkpoint, path)


def load_weight(path):
    root = 'result/checkpoint'
    path = os.path.join(root, path)
    checkpoint = torch.load(path)

    return checkpoint
