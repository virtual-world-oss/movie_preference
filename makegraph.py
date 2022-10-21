import json

import networkx
import numpy as np
from utils import cal_num_tags

def tagIdToItsIdInGraph(id):
    return id+4044

def movieIdToItsIdInGraph(id):
    return id + 2022

def userIdToItsIdInGraph(id):
    return id

def str2tupleforkey(key):
    key = key.strip('"')
    key = key.lstrip('(')
    key = key.rstrip(')')
    key = key.split(',')
    key = [int(id) for id in key]
    return key

def makegraph():
    num_user = 2022
    num_movie = 2022
    num_tag,tagid = cal_num_tags()
    graph_shape = num_tag+num_user+num_movie
    adj = np.zeros((graph_shape,graph_shape))#0-2021 user; 2022-4043,movie; 4044-7136,tag.
    # print(adj.shape)

    # liu:这部分设置movie跟tag之间的边，包含关系
    with open('data/train/movie.json') as f:
        movie2tag = json.load(f)
        # print(len(movie2tag))
        for item in movie2tag:
            taglist = movie2tag[item]
            movieid = int(item)
            # print(taglist,movieid)
            for _ in range(len(taglist)):
                tagid = taglist[_]
                weight = 1 - _*0.05
                adj[movieIdToItsIdInGraph(movieid)][tagIdToItsIdInGraph(tagid)] = weight

    # liu:用户跟tag之间的边，这是用户一定喜欢的tag的边
    with open('data/train/user2tagslike.json') as f1:
        user2tagslike = json.load(f1)
        # print(len(user2tagslike))
        for item in user2tagslike:
            # break
            userid = userIdToItsIdInGraph(int(item))
            tagidlist = user2tagslike[item]
            assert -1 not in tagidlist
            # print(userid,tagidlist)
            # break
            for tagid in tagidlist:
                tagid = tagIdToItsIdInGraph(tagid)
                adj[userid][tagid] = 10



    with open('data/train/user2movie2rating.json') as f2:
        obs = json.load(f2)
        for item in obs:
            # print(item)
            # key = str2tupleforkey(item)
            # userid = key[0]
            # movieid = key[1]
            # # print(userid,movieid)
            # userid = userIdToItsIdInGraph(userid)
            # movieid = movieIdToItsIdInGraph(movieid)
            # adj[userid][movieid] = 1
            movies = obs[item]
            userid = userIdToItsIdInGraph(int(item))
            for movie in movies:
                movieid = movieIdToItsIdInGraph(int(movie))
                rating = movies[movie]
                adj[userid][movieid] = rating

    for i in range(graph_shape):
        for j in range(graph_shape):
            if adj[i][j] == 0 and adj[j][i] == 0:
                continue
            else:
                adj[i][j] = 1
                adj[j][i] = 1

    # for i in range(graph_shape):
    #     assert adj[i][i] == 0
    # num_edges = np.sum(adj == 1)
    # print(num_edges,graph_shape*graph_shape)
    outfile = 'data/train/adj.npy'
    np.save(outfile, adj)

def init_feature():
    num_user = 2022
    num_movie = 2022
    num_tag, tagid = cal_num_tags()
    graph_shape = num_tag + num_user + num_movie
    # print(graph_shape)
    features = np.eye(graph_shape)
    # print(features.shape)
    # print(np.sum(features == 1))
    out_file = 'data/train/features.npy'
    np.save(out_file,features)

if __name__ == '__main__':

    makegraph()
    # out_file = 'data/train/features.npy'
    # features = np.load(out_file)
    # print(features.shape)
    # init_feature()
    # x = np.array([[1,0],[0,1]])
    # y = np.sum(x == 1)
    # print(y)
    # outfile = 'data/train/adj.npy'
    # adj = np.load(outfile)
    #
    # adj_shape = adj.shape[0]
    #
    # num_independent_node = 0
    # for i in range(adj_shape):
    #     x = False
    #     for j in range(adj_shape):
    #         if adj[i][j] == 1:
    #             x = True
    #     if x == False:
    #         num_independent_node += 1
    #
    # print(num_independent_node)




    pass