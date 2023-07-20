import random
import torch
import numpy as np
import scipy.ndimage
from scipy.signal import medfilt
from scipy.spatial.distance import cdist


random.seed(0)


def zoom(p, target_length=32, joints_num=20, joints_dim=3):
    length = p.shape[0]
    p_new = np.empty([target_length, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = scipy.ndimage.zoom(p[:, m, n], target_length / length)[:target_length]
    return p_new


def temporal_padding(pose, target_len):
    t, *target_shape = pose.shape

    if t >= target_len:
        start = random.randint(0, t-target_len)
        end = start + target_len
        return pose[start:end]

    target = torch.zeros(target_len, *target_shape)
    start = random.randint(0, target_len - t)
    end = start + t
    target[start:end] = pose

    return target


def spatial_padding(pose):
    if pose.shape == (42, 3):
        return pose

    target = torch.zeros(42, 3)
    try:
        target[:21] = pose
    except RuntimeError:
        #print(pose.shape)
        raise RuntimeError

    return target


class Config:
    def __init__(self, n_joints, joint_dims, n_frames):
        self.n_joints = n_joints
        self.joint_dims = joint_dims
        self.n_frames = n_frames


if __name__ == '__main__':
    import torch

    seq_length = 8
    target_length = 4
    n_joints = 22
    joint_dims = 4

    c = Config(n_joints, joint_dims, target_length)

    pose = np.arange(seq_length * n_joints * joint_dims).reshape((seq_length, n_joints, joint_dims))
    # pose = np.random.randn(5, n_joints, joint_dims).round(decimals=4)
    pose_new = get_CG(pose, c)
