import os
import random
import pickle
from pathlib import Path
import torch
import time
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

import data_processing as dp


random.seed(0)
torch.set_default_dtype(torch.float64)


# New dataset with additional 6 pose keypoints
class MyDataset(Dataset):
    def __init__(self, split, root, n_joints, n_frames, n_classes, is_balanced):
        self.split = split
        self.root = root
        self.n_joints = n_joints
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.is_balanced = is_balanced
        self.data = self.make_dataset()

    def __getitem__(self, index):
        vid_name, label, start, end = self.data[index]

        pose = self.extract_pose(vid_name, start, end)
        pose = dp.temporal_padding(pose, self.n_frames)
        motion = self.get_CG(pose)

        data = [pose, motion]

        return data, label

    def make_dataset(self):

        # Load data to RAM to avoid multiple loading
        self.pose_dict = {}
        pose_path = Path(os.path.join(self.root, 'pose_new'))

        for pose_file in pose_path.glob('*.npy'):
            subj_name = pose_file.stem
            pose_data = np.load(pose_file.as_posix(), allow_pickle=True)
            self.pose_dict[subj_name] = pose_data

        # Create per-line annotation
        data = []
        annot_path = os.path.join(self.root, 'annot', f'{self.split}_list.txt')
        with open(annot_path, 'r') as f:
            for line in f:
                vid_name, label, cls_name, start, end, hand = line.split()
                label, start, end = int(label), int(start), int(end)

                if self.is_balanced and label == 0:
                    random_number = random.randint(0, 20)
                    if random_number != 0:
                        continue
                
                label = torch.tensor(label)
                data.append([vid_name, label, start, end])
        return data
    
    def extract_pose(self, vid_name, start, end):
        pose = self.pose_dict[vid_name][start-1: end]
        pose = pose[:, :self.n_joints, :]
        return torch.tensor(pose)

    def get_CG(self, p):
        M = []
        iu = np.triu_indices(self.n_joints, 1, self.n_joints)
        for f in range(self.n_frames):
            #distance max
            d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, 3])]),'euclidean')
            d_m = d_m[iu]
            M.append(d_m)
        M = np.stack(M)
        return M

    def __len__(self):
        return len(self.data)


# Old datase for old format of .npy files
class OldDataset(Dataset):
    def __init__(self, split, root, n_frames, n_joints, n_classes, is_balanced):
        self.split = split
        self.root = root
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.is_balanced = is_balanced
        self.data = self.make_dataset()

    def __getitem__(self, index):
        vid_name, label, start, end = self.data[index]
        pose = self.extract_pose(vid_name, start, end)
        pose = dp.temporal_padding(pose, self.n_frames)
        motion = self.get_CG(pose)
        data = [pose, motion]

        return data, label
    
    def make_dataset(self):
        self.pose_dict = {}
        pose_path = Path(self.root) / 'pose_ver1'

        for pose_file in pose_path.glob('*.npy'):
            subj_name = pose_file.stem
            pose_data = np.load(pose_file.as_posix(), allow_pickle=True).item()
            self.pose_dict[subj_name] = pose_data

        data = []
        annot_path = os.path.join(self.root, 'annot', f'{self.split}_list.txt')
        with open(annot_path, 'r') as f:
            for line in f:
                vid_name, label, cls_name, start, end, hand = line.split()
                label, start, end = int(label), int(start), int(end)

                if self.is_balanced and label == 0:
                    random_number = random.randint(0, 20)
                    if random_number != 0:
                        continue

                label = torch.tensor(label)
                data.append([vid_name, label, start, end])

        return data

    def extract_pose(self, vid_name, start, end):
        pose_shape = (42, 3)
        pose_final = []

        data = self.pose_dict[vid_name]

        for idx in range(start, end+1):
            try:
                pose = data[idx]
                pose = torch.tensor(pose).reshape(-1, 3)
                pose = dp.spatial_padding(pose)
            except KeyError:
                pose = torch.zeros(pose_shape)

            pose_final.append(pose)
        pose_final = torch.stack(pose_final)
        
        return pose_final

    def get_CG(self, p):
        #print(p.shape)
        M = []
        iu = np.triu_indices(42, 1, 42)
        for f in range(self.n_frames):
            d_m = cdist(p[f],np.concatenate([p[f],np.zeros([1, 3])]),'euclidean')
            d_m = d_m[iu]
            M.append(d_m)
        M = np.stack(M)
        return M

    def __len__(self):
        return len(self.data)


class IPN(Dataset):

    def __init__(self, split, root, n_frames, sampling_rate):
        self.split = split
        self.root = root
        self.n_frames = n_frames
        self.sampling_rate = sampling_rate
        self.data = self.make_dataset()

    def __getitem__(self, index):
        vid_name, label, start, end = self.data[index]
        pose = self.extract_pose(vid_name, start, end)

        pose = dp.temporal_padding(pose, self.n_frames)
        label = dp.temporal_padding(label, self.n_frames)

        return pose, label

    def make_dataset(self):
        data = []
        annot_file = f'{self.split}listpose.txt'
        # annot_dir = Path(root) / annot_file
        annot_dir = os.path.join(self.root, annot_file)

        with open(annot_dir, 'r') as f:
            for line in f:
                vid_name, label, start, end = line.split()
                label, start, end = int(label), int(start), int(end)
                label = torch.ones(end - start + 1) * label
                data.append([vid_name, label, start, end])

        return data

    def extract_pose(self, vid_name, start, end):
        pose_shape = (21, 3)
        path = Path(self.root) / 'pose' / f'{vid_name}.pickle'

        with path.open('rb') as f:
            data = pickle.load(f)

        pose_final = []

        for line in data[start:end+1]:
            pose = line['pose3D']

            # If pose is None, get pose from previous frame
            # If prev_pose is None, set to zero tensor
            prev_pose = pose_final[-1] if pose_final != [] \
                                        else torch.zeros(*pose_shape)
            if pose == []:
                pose = prev_pose

            t_pose = torch.tensor(pose)
            t_pose = dp.spatial_padding(t_pose)

            pose_final.append(t_pose)
        pose_final = torch.stack(pose_final)

        return pose_final

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = MyDataset('train', 'data/', 32, 0)
    pose, label = ds[1]

    print(pose.shape)
    print(label)
