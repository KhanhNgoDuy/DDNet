import numpy as np
import argparse
import random
import torch
import csv
from time import perf_counter as pc

from dataset import IPN, MyDataset
from model.TD_Net import TDNet
from utils import *
import data_processing as dp


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


def main():
    root='data/MyDataset'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_frames = 60
    n_joints = 42
    joint_dims = 3
    n_filters = 64
    feat_dims = 861
    n_classes = 20 
    drop_rate = 0.5

    params = [n_joints,
              joint_dims,
              feat_dims,
              n_filters,
              n_frames,
              n_classes,
              drop_rate]

    model = TDNet(*params)
    model.eval().to(device)

    pose_dict, gt_dict = get_data(root)
    
    video_list = []
    with open(os.path.join(root, 'annot/val_list.txt'), 'r') as f:
        for line in f:
            name = line.split()[0]
            if name in video_list:
                continue
            video_list.append(name)

    results = {}
    # video contains (pose, ground truth)
    for video in video_list:
        pose = pose_dict[video]
        gt = gt_dict[video]
        probs, targets = run_online(model, pose, gt, n_frames)
        results['video'] = {'prob': probs, 'target': targets}


def run_online(model, pose, gt, window_size):
    vid_length = gt.shape

    # The input pose is defined to be the pose from the last `window_size` frames
    # The label is defined to be the label of the current frame
    prob_ls = []
    output_ls = []
    correct = false = 0

    with torch.no_grad():
        for i in range(vid_length):
            if i < window_size:
                x = torch.zeros(window_size, 42, 3)
                x[-(i+1): 0] = pose[0: i+1]
            else:
                x = pose[i - window_size: i]
            x = x.unsqueeze(0)
            y = gt[i].unsqueeze(0)

            prob = model(x, get_CG(x))
            output = torch.argmax(prob, dim=1)

            correct += (output == y).sum()
            false += (output != y).sum()
            prob_ls.append(prob)
            output_ls.append(output)

    print(f"Corrected: {correct}/{vid_length}")
    print(f"Accuracy: {correct / vid_length}")
    





def get_data(root):
    annot_path = os.path.join(root, 'annot/val_list.txt')
    pose_path = os.path.join(root, 'pose')
    len_path = os.path.join(root, 'annot/n_frames.csv')

    with open(len_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        len_dict = dict(reader)

    # get ground_truth per frame
    ground_truth = {}

    for key in len_dict.keys():
        ground_truth[key] = torch.zeros(int(len_dict[key]))     # Create an empty array of (t,)

    with open(annot_path, 'r') as f:
        for line in f:
            name, label, _, start, end, _ = line.split()
            label, start, end = int(label), int(start), int(end)
            ground_truth[name][start-1:end] = label    

    # get pose per frame
    pose = {}

    for name in ground_truth.keys():
        pose[name] = torch.zeros((int(len_dict[name]), 42, 3))    # Create an empty array of (t, 42, 3)

    for pose_file in os.listdir(pose_path):
        name, ext = os.path.splitext(pose_file)
        p_dict = np.load(os.path.join(pose_path, pose_file), allow_pickle=True).item()
        
        p_tensor = []
        for idx, p in enumerate(p_dict.values()):
            p = torch.tensor(p).reshape(-1, 3)
            p = dp.spatial_padding(p)
            p_tensor.append(p)
        p_tensor = torch.stack(p_tensor)

        pose[name][list(p_dict.keys())] = p_tensor

    return pose, ground_truth


if __name__ == '__main__':
    main()