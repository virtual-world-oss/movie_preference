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
from makegraph import userIdToItsIdInGraph,tagIdToItsIdInGraph,movieIdToItsIdInGraph
cofig = {
    'device':0,#0 for cpu and 1 for cuda
    'epochs':500,
    'lr':0.01,
    'weight_decay':5e-4,
    'nhid1':16,
    'nhid2':16,
    'nhid3':16,
    'nclass':2,#取1则使用sigmoid，取2则使用softmax
    'dropout':0.5,
    'seed':3407,
    'epoch_save':50
}

file_name = 'Model/ModelSave/GCNModel_train_acc_0.7891347566773274_val_acc_0.7956431535269709_epoch_200.pth'
model = torch.load(file_name)
model.eval()
import pandas as pd
file_name = 'data/test/Phase1_test_dataset.csv'
test_file = pd.read_csv(file_name)
test_data = test_file.values.tolist()
# print(test_data)
true_test_list = []
for z in test_data:
    # print(z)
    tmp = [userIdToItsIdInGraph(z[0]),tagIdToItsIdInGraph(z[1])]
    true_test_list.append(tmp)

# print(true_test_list)
adj,feature,idx_train,idx_train_label,idx_val,idx_val_label,idx_test,idx_test_label = load_data(whether_negative_sample=True,label_type=cofig['nclass'],seed = cofig['seed'])
embedding = model.encode(adj, feature)
output = model.decode(embedding, true_test_list)
print(output)
out = output[0]
print(output.shape,out.shape)
pro = []
for out in output:
    pro.append(torch.exp(out[1]).item())
print(pro)
test_file.insert(loc=2,column='',value=pro)
print(test_file)
test_file.to_csv('submission.csv',index=False)