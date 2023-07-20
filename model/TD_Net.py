import torch
import torch.nn as nn
# from torchinfo import summary
from model.module import c1d, d1d, SpatialDropout1d, MaxPool1d, PoseMotion, c1dBlock, GlobalMaxPool1d


class SlowBlock(nn.Module):
    def __init__(self, in_channels, n_filters, drop_rate):
        super(SlowBlock, self).__init__()

        self.c1d_1 = c1d(in_channels, n_filters*2, 1)
        self.c1d_2 = c1d(n_filters*2, n_filters, 3)
        self.c1d_3 = c1d(n_filters, n_filters, 1)
        self.max_pool_3 = MaxPool1d(2)
        self.drop_out = SpatialDropout1d(drop_rate)
    
    def forward(self, x):
        x = self.drop_out(self.c1d_1(x))
        x = self.drop_out(self.c1d_2(x))
        x = self.c1d_3(x)
        x = self.max_pool_3(x)
        x = self.drop_out(x)

        return x


class FastBlock(nn.Module):
    def __init__(self, in_channels, n_filters, drop_rate):
        super().__init__()

        self.conv_1 = c1d(in_channels, n_filters*2, 1)
        self.conv_2 = c1d(n_filters*2, n_filters, 3)
        self.conv_3 = c1d(n_filters, n_filters, 1)
        self.drop_out = SpatialDropout1d(drop_rate)
    
    def forward(self, x):
        x = self.drop_out(self.conv_1(x))
        x = self.drop_out(self.conv_2(x))
        x = self.drop_out(self.conv_3(x))

        return x


class JCDBlock(nn.Module):
    def __init__(self, in_channels, n_filters, drop_rate):
        super().__init__()

        self.conv_1 = c1d(in_channels, n_filters*2, 1)
        self.conv_2 = c1d(n_filters*2, n_filters, 3)
        self.conv_3 = c1d(n_filters, n_filters, 1)
        self.max_pool_3 = MaxPool1d(2)
        self.drop_out = SpatialDropout1d(drop_rate)
    
    def forward(self, x):
        x = self.drop_out(self.conv_1(x))
        x = self.drop_out(self.conv_2(x))
        x = self.conv_3(x)
        x = self.max_pool_3(x)
        x = self.drop_out(x)

        return x


class TDNet(nn.Module):
    def __init__(self, n_joints, joint_dims, feat_dims, n_filters, n_frame, n_classes, drop_rate):
        super().__init__()

        self.pose_motion = PoseMotion(n_frame)
        self.slow_block = SlowBlock(n_joints*joint_dims, n_filters, drop_rate)
        self.fast_block = FastBlock(n_joints*joint_dims, n_filters, drop_rate)
        self.JCD_block = JCDBlock(feat_dims, n_filters, drop_rate)

        self.block_1 = nn.Sequential(
                          c1dBlock(n_filters*3, n_filters*2),
                          MaxPool1d(2),
                          SpatialDropout1d(drop_rate),
                          c1dBlock(n_filters*2, n_filters*4),
                          MaxPool1d(2),
                          SpatialDropout1d(drop_rate),
                          c1dBlock(n_filters*4, n_filters*8),
                          SpatialDropout1d(drop_rate)
        )

        self.block_2 = nn.Sequential(
                          GlobalMaxPool1d(),
                          d1d(n_filters*8, 128),
                          nn.Dropout(0.5),
                          d1d(128, 128),
                          nn.Dropout(0.5),
                          nn.Linear(128, n_classes),
                          # nn.Softmax(dim=1)
        )

    def forward(self, P, M):
        diff_slow, diff_fast, x_1 = self.pose_motion(P)

        x = self.JCD_block(M)
        x_d_slow = self.slow_block(diff_slow)
        x_d_fast = self.fast_block(diff_fast)

        x = torch.cat([x, x_d_slow, x_d_fast], dim=-1)
        x = self.block_1(x)
        x = self.block_2(x)
        
        return x


if __name__ == '__main__':
    batch_size = 16
    n_frames = 32
    n_joints = 48
    joint_dims = 3
    n_filters = 16
    feat_dims = 210
    n_classes = 14
    drop_rate = 0.5
    training = True

    params = [n_joints,
              joint_dims,
              feat_dims,
              n_filters,
              n_frames,
              n_classes,
              drop_rate,
              training]

    model = TDNet(*params)
    summary(model, input_size=[(batch_size, n_frames, n_joints, joint_dims), (batch_size, n_frames, feat_dims)])
