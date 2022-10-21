import torch
import torch.nn as nn
import torch.nn.functional as F
from GCNModel.layers import GraphConvolution
from Model.encoder import GCNencoder
from Model.decoder import Lineardecoder

class Recommender(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2,nhid3, nclass, dropout):
        super(Recommender, self).__init__()

        self.encoder = GCNencoder(nfeat,nhid1,nhid2,dropout)

        self.decoder = Lineardecoder(nhid2,nhid3,nclass,dropout)

    def set_embedding(self,vocab_size,embed_dim):
        self.embedding = nn.Embedding(vocab_size,embed_dim)


    def encode(self,adj,feature):
        ids = list(range(feature.shape[0]))
        ids = torch.tensor(ids)
        in_feat = self.embedding(ids)
        # return self.encoder(feature, adj)
        return self.encoder(in_feat,adj)

    def decode(self,feature,idx):
        feature_in = torch.Tensor([])
        for item in idx:
            user_emb = feature[item][0]
            # print(f'user_emb：{user_emb}')

            tag_emb = feature[item][1]
            # print(f'tag+emb:{tag_emb}')
            now_feature = torch.cat([user_emb,tag_emb],-1)
            now_feature = now_feature.unsqueeze(0)
            feature_in = torch.cat([feature_in,now_feature],0)
            # print(feature_in)
        return self.decoder(feature_in)

