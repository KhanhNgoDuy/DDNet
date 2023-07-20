import os
import sys
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from model.TD_Net import TDNet
from utils import *


torch.set_default_dtype(torch.float64)

def main(model_folder, model_name):
    print()

    path = 'result/checkpoint/'
    #model_folder = 'd_60_64'
    #model_name = 'acc_162.pth'
    name = model_name.split('.')[0]
    save_dir = os.path.join('result/logging', model_folder, name)
    print(save_dir)
    if os.path.exists(save_dir):
        print(f'Skipping {save_dir}', end='\n\n')
    #    return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    checkpoint_path = os.path.join(path, model_folder, model_name)
    n_frames, n_filters = model_folder.split('_')[-2:]
    n_frames, n_filters = int(n_frames), int(n_filters)
    
    datasets = get_datasets('Custom', n_frames, 20, 0, 'data', False)
    dataloaders = get_dataloaders(1, datasets, 0)

    model = TDNet(42, 3, 861, n_filters, n_frames, 20, 0.5).float().to(device)
    weights = torch.load(checkpoint_path, map_location=device)['model']
    model.load_state_dict(weights)
    #model.eval()

    pred, gt = forward(model, dataloaders['val'], device)

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    result = np.stack([pred.numpy(), gt.numpy()], axis=0)
    np.save(os.path.join(save_dir, 'result.npy'), result)

    pred_19 = pred[gt != 0]
    gt_19 = gt[gt != 0]

    report = classification_report(gt, pred)
    with open(os.path.join(save_dir, 'report.txt'), 'w') as f:
        f.write(report)
    
    cm = confusion_matrix(gt, pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(save_dir, 'cm.jpg'), dpi=300)
    #plt.show()

    cm = confusion_matrix(gt_19, pred_19)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(save_dir, 'cm19.jpg'), dpi=300)
    #plt.show()


def forward(model, dataloader, device):
    print('Evaluating...')
    model.eval()

    with torch.no_grad():
        preds = []
        gts = []
        correct = 0

        for [pose, motion], target in tqdm(dataloader, total=len(dataloader)):
            pose, motion, target = pose.float().to(device), motion.float().to(device), target.float().to(device)
            output = model(pose, motion)
            #pred = torch.argmax(output, dim=1)[0]
            
            pred = torch.argmax(output, dim=1)
            correct += (pred == target).sum()

            #raise SyntaxError
            preds.append(pred)
            gts.append(target[0])
    
    print(f'Accuracy: {correct/len(dataloader)} ({correct}/{len(dataloader)})')
    preds = torch.tensor(preds)
    gts = torch.tensor(gts)

    return preds, gts


if __name__ == '__main__':
    model_folder = sys.argv[1]
    model_name = sys.argv[2]
    main(model_folder, model_name)
