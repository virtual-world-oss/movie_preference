import torch.nn as nn
import torch.nn.functional as F

class Lineardecoder(nn.Module):
    def __init__(self, infeat, nhid, nclass, dropout):
        super(Lineardecoder, self).__init__()

        self.nclass = nclass
        self.fc1 = nn.Linear(in_features=2*infeat,out_features=nhid)
        self.fc2 = nn.Linear(in_features=nhid,out_features=nclass)
        self.dropout = dropout

    def forward(self,feature):
        x = F.relu(self.fc1(feature))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)

        if self.nclass == 1:
            return F.sigmoid(x)
        elif self.nclass == 2:
            return F.log_softmax(x, dim=1)