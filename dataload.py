import json
import random

import scipy.sparse as sp
import numpy as np
import torch
from makegraph import tagIdToItsIdInGraph,userIdToItsIdInGraph,movieIdToItsIdInGraph

def load_data(whether_negative_sample = False,train_ratio = 0.8,val_ratio = 0.1,test_ratio = 0.1,label_type = 2,seed = 0):
    adj = np.load('data/train/adj.npy')
    # feature = np.load('data/train/features.npy')
    adj = normalize(adj + np.eye(adj.shape[0]))
    np.random.seed(seed)
    feature = np.random.randn(adj.shape[0],adj.shape[0])
    feature = normalize(feature)

    with open('data/train/user2tagslike.json') as liketag:
        user2tagslike = json.load(liketag)
    with open('data/train/user2tagsdislike.json') as disliketag:
        user2tagsdislike = json.load(disliketag)


    all_pos_sample = []
    all_neg_sample = []
    if not whether_negative_sample:
        for userid in user2tagslike:
            tripulet = [[userIdToItsIdInGraph(int(userid)),tagIdToItsIdInGraph(x),1] for x in user2tagslike[userid]]
            # print(tripulet)
            all_pos_sample.extend(tripulet)
            # print(all_pos_sample)
        random.shuffle(all_pos_sample)
        # print(all_pos_sample)
        # print(len(all_pos_sample))
        all_len = len(all_pos_sample)
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)
        all_idx = []
        label = []
        for item in all_pos_sample:
            all_idx.append(item[:2])
            label.append(item[2])
        # print(len(all_idx),len(label))
        idx_train = all_idx[:train_len]
        idx_val = all_idx[train_len:train_len+val_len]
        idx_test = all_idx[train_len+val_len:]

        idx_train_label = label[:train_len]
        idx_val_label = label[train_len:train_len+val_len]
        idx_test_label = label[train_len+val_len:]

        # print(len(idx_train),len(idx_train_label),len(idx_val),len(idx_val_label),len(idx_test),len(idx_test_label))
        # print(idx_train[len(idx_train)-1],idx_val[0])
        adj = torch.FloatTensor(adj)
        feature = torch.FloatTensor(feature)

        if label_type == 1:
            idx_train_label = torch.FloatTensor(idx_train_label)
            idx_val_label = torch.FloatTensor(idx_val_label)
            idx_test_label = torch.FloatTensor(idx_test_label)
        elif label_type == 2:
            idx_train_label = torch.LongTensor(idx_train_label)
            idx_val_label = torch.LongTensor(idx_val_label)
            idx_test_label = torch.LongTensor(idx_test_label)
    else:
        for userid in user2tagslike:
            tripulet = [[userIdToItsIdInGraph(int(userid)),tagIdToItsIdInGraph(x), 1] for x in user2tagslike[userid]]
            # print(tripulet)
            all_pos_sample.extend(tripulet)
        for userid in user2tagsdislike:
            neg_tripulet = [[userIdToItsIdInGraph(int(userid)),tagIdToItsIdInGraph(x), 0] for x in user2tagsdislike[userid]]
            all_neg_sample.extend(neg_tripulet)

        all_sample = all_pos_sample + all_neg_sample
        random.shuffle(all_sample)
        all_len = len(all_sample)
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)
        all_idx = []
        label = []
        for item in all_sample:
            all_idx.append(item[:2])
            label.append(item[2])
        idx_train = all_idx[:train_len]
        idx_val = all_idx[train_len:train_len + val_len]
        idx_test = all_idx[train_len + val_len:]

        idx_train_label = label[:train_len]
        idx_val_label = label[train_len:train_len + val_len]
        idx_test_label = label[train_len + val_len:]

        # print(len(idx_train),len(idx_train_label),len(idx_val),len(idx_val_label),len(idx_test),len(idx_test_label))
        # print(idx_train[len(idx_train)-1],idx_val[0])
        adj = torch.FloatTensor(adj)
        feature = torch.FloatTensor(feature)

        if label_type == 1:
            idx_train_label = torch.FloatTensor(idx_train_label)
            idx_val_label = torch.FloatTensor(idx_val_label)
            idx_test_label = torch.FloatTensor(idx_test_label)
        elif label_type == 2:
            idx_train_label = torch.LongTensor(idx_train_label)
            idx_val_label = torch.LongTensor(idx_val_label)
            idx_test_label = torch.LongTensor(idx_test_label)


    return adj,feature,idx_train,idx_train_label,idx_val,idx_val_label,idx_test,idx_test_label

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

if __name__ == '__main__':
    adj,feature,idx_train,idx_train_label,idx_val,idx_val_label,idx_test,idx_test_label = load_data(whether_negative_sample=True)
    # print(torch.sum(idx_train_label == 0) + torch.sum(idx_val_label == 0) + torch.sum(idx_test_label == 0))
    # print(idx_train_label.shape)
    neg_num = 0
    for label in idx_train_label:
        if label.item() == 0:
            neg_num+=1
    for label in idx_val_label:
        if label.item() == 0:
            neg_num+=1
    for label in idx_test_label:
        if label.item() == 0:
            neg_num+=1
    print(neg_num)
    # print(len(idx_val),len(idx_test),len(idx_train))
