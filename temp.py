import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch


# root = 'result/logging'
# acc_dict = {}
# matching_folders = Path(root).glob('d_70_*')

# for folder in matching_folders:
#     accs = []
#     window_size = folder.as_posix().split('_')[-1]
#     if window_size not in ["80", "96"]:
#         continue

#     log_file = (folder / "log_loss.json").as_posix()
#     with open(log_file, 'r') as f:
#         log = json.load(f)
#     for val in log.values():
#         accs.append(val['Val accuracy'])
#     acc_dict[window_size] = accs

# for window_size in acc_dict.keys():
#     acc = acc_dict[window_size]
#     plt.plot(acc, label=window_size)

# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy with different filters")

# plt.legend()
# plt.savefig("num-filters.jpg", dpi=300, bbox_inches='tight')
# plt.show()
        

from model.TD_Net import TDNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n_joints = 48
joint_dims = 3
feat_dims = 1128
n_frames = 60
n_class = 20
drop_rate = 0.5


n_filters = 40
net_40 = TDNet(n_joints, joint_dims, feat_dims, n_filters, n_frames, n_class, drop_rate)
print('40', count_parameters(net_40))

n_filters = 64
net_64 = TDNet(n_joints, joint_dims, feat_dims, n_filters, n_frames, n_class, drop_rate)
print('64', count_parameters(net_64))

n_filters = 80
net_80 = TDNet(n_joints, joint_dims, feat_dims, n_filters, n_frames, n_class, drop_rate)
print('80', count_parameters(net_80))

n_filters = 96
net_96 = TDNet(n_joints, joint_dims, feat_dims, n_filters, n_frames, n_class, drop_rate)
print('96', count_parameters(net_96))

n_filters = 128
net_128 = TDNet(n_joints, joint_dims, feat_dims, n_filters, n_frames, n_class, drop_rate)
print('128', count_parameters(net_128))
