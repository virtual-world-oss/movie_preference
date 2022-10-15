import torch.nn as nn
import torch.nn.functional as F
from GCNModel.layers import GraphConvolution

class GCNencoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(GCNencoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x