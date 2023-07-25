import random
import torch
import numpy as np
import scipy.ndimage
from scipy.signal import medfilt
from scipy.spatial.distance import cdist


random.seed(0)


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
    target[:21] = pose

    return target

