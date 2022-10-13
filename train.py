from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from GCNModel.utils import load_data, accuracy
from GCNModel.models import GCN
from dataload import load_data
from Model.model import Recommender

'''
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
                    '''


adj,feature,idx_train,idx_train_label,idx_val,idx_val_label,idx_test,idx_test_label = load_data()
cofig = {
    'device':0,#0 for cpu and 1 for cuda
    'epochs':5,
    'lr':0.01,
    'weight_decay':5e-4,
    'nhid1':16,
    'nhid2':16,
    'nhid3':16,
    'nclass':2,
    'dropout':0.5
}

model = Recommender(feature.shape[1],cofig['nhid1'],cofig['nhid2'],cofig['nhid3'],cofig['nclass'],cofig['dropout'])

optimizer = optim.Adam(model.parameters(),
                       lr=cofig['lr'], weight_decay=cofig['weight_decay'])

if cofig['device'] != 0:
    model.cuda()
    features = feature.cuda()
    adj = adj.cuda()


# def acc(out,label):
#     preds = out.max(1)[1]
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):
    #train
    t = time.time()
    model.train()
    optimizer.zero_grad()
    embedding = model.encode(adj,feature)
    output = model.decode(embedding,idx_train)
    loss_train = F.nll_loss(output,idx_train_label)
    acc_train = accuracy(output, idx_train_label)
    loss_train.backward()
    optimizer.step()

    #val
    model.eval()
    embedding = model.encode(adj, feature)
    output = model.decode(embedding, idx_val)
    loss_val = F.nll_loss(output, idx_val_label)
    acc_val = accuracy(output, idx_val_label)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    # output = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    # loss_train.backward()
    # optimizer.step()
    #
    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)
    #
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    embedding = model.encode(adj, feature)
    output = model.decode(embedding, idx_test)
    print(output)
    loss_test = F.nll_loss(output, idx_test_label)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output, idx_test_label)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item())
          )
          # "accuracy= {:.4f}".format(acc_test.item()))

t_total = time.time()
for epoch in range(cofig['epochs']):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test()