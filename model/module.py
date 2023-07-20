import torch
import torch.nn as nn
import torch.nn.functional as F


class c1d(nn.Module):
    # Input shape: 4D (batch_size, n_frames, n_joints, joint_dims)
    # Input shape: 3D (batch_size, n_frames, feat_dims)     # JCD Branch
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.out_dims = out_channels
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding='same',
                                bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        batch_size, time, dim = x.shape[0], x.shape[-2], x.shape[-1]

        x = x.view(-1, time, dim).permute(0, 2, 1)  # Keep only the last 2 dims
        x = self.conv1d(x)

        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)
        x = x.permute(0, 2, 1).view(batch_size, -1, time, self.out_dims).squeeze(1)

        return x


class d1d(nn.Module):
    # Input shape: (batch, n_dims)
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dense = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.dense(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x


class SpatialDropout1d(nn.Module):
    # Input shape: (batch_size, n_frames, n_dims)   (JCD Branch)
    def __init__(self, drop_rate):
        super().__init__()
        self.dropout1d = nn.Dropout1d(p=drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.dropout1d(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x


class PoseMotion(nn.Module):
    # Input shape: 4D (batch_size, n_frames, n_joints, joint_dims)
    def __init__(self, n_frame):
        super().__init__()
        self.n_frame = n_frame

    def forward(self, P):
        frame_l = self.n_frame

        P_slow = self.pose_diff(P)
        batch, n_frame, _, _ = P_slow.shape
        P_slow = P_slow.reshape(batch, frame_l, -1)

        P_fast = P[:, ::2, ...]
        P_fast = self.pose_diff(P_fast)
        batch, n_frame, _, _ = P_fast.shape
        P_fast = P_fast.reshape(batch, int(frame_l / 2), -1)

        batch, n_frame, _, _ = P.shape
        x = P.reshape(batch, frame_l, -1)

        return P_slow, P_fast, x

    def pose_diff(self, x):
        # With shape B, N_frame, N_joint, D
        h, w = x.shape[1], x.shape[2]
        x = x[:, 1:, ...] - x[:, :-1, ...]
        # x = x.permute(2, 0, 1)     # For C, H, W input
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return x


class c1dBlock(nn.Module):
    # Input shape: 3D (batch_size, n_frames/N, n_filters)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1d_1 = c1d(in_channels, out_channels, 3)
        self.c1d_2 = c1d(out_channels, out_channels, 3)

    def forward(self, x):
        x = self.c1d_1(x)
        x = self.c1d_2(x)
        return x


class MaxPool1d(nn.Module):
    # Input shape: 3D (batch_size, n_frames/N, n_filters)
    def __init__(self, kernel_size):
        super().__init__()
        self.max_pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.max_pool(x)
        x = x.permute(0, 2, 1)
        return x


class GlobalMaxPool1d(nn.Module):
    # Input shape: 3D (batch, n_frames/N, n_filters)
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.global_max_pool(x).squeeze(-1)

        return x


if __name__ == '__main__':
    batch_size = 16
    n_frames = 32
    n_joints = 21
    joint_dims = 3
    out_channels = 100
    kernel_size = 2

    # conv1d = c1d(joint_dims, out_channels, kernel_size)
    # dense1d = d1d(joint_dims, out_channels)
    # drop_out = SpatialDropout1d(0.5, True)
    # pose_motion = PoseMotion(n_frames)
    # block = c1dBlock(joint_dims, out_channels)
    max_pool = MaxPool1d(kernel_size)

    x2d = torch.randn((batch_size, joint_dims))
    x3d = torch.randn((batch_size, n_joints, joint_dims))
    x4d = torch.randn((batch_size, n_frames, n_joints, joint_dims))

    x = x3d
    y = max_pool(x)
    print(x.shape)
    print(y.shape)
