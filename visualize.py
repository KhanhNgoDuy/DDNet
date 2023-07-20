import os
import sys
import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_loss(path, mode):
    file_name = os.path.join(path, 'log_loss.json')
    with open(file_name, 'r') as f:
        data = json.load(f)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for key in data.keys():
        train_loss.append(data[key]['Train loss'])
        val_loss.append(data[key]['Val loss'])
        train_acc.append(data[key]['Train accuracy'])
        val_acc.append(data[key]['Val accuracy'])

    if mode == 'loss':
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.savefig('loss.jpg', dpi=300)
    elif mode == 'acc':
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.savefig('acc.jpg', dpi=300)
    elif mode == 'all':
        train_loss = torch.tensor(train_loss)
        val_loss = torch.tensor(val_loss)
        train_acc = torch.tensor(train_acc)
        val_acc = torch.tensor(val_acc)
    	
        train_loss /= train_loss.max()
        val_loss /= val_loss.max()
        train_acc /= 100
        val_acc /= 100
    	
    	#plt.plot(train_loss)
        plt.plot(val_loss)
    	#plt.plot(train_acc)
        plt.plot(val_acc)
    plt.show()


if __name__ == '__main__':
    root = 'result/logging'
    name = sys.argv[1]
    mode = sys.argv[2]

    path = os.path.join(root, name)
    plot_loss(path, mode)
