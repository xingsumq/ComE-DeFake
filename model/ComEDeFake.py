from torch import nn
from model.layers import HGNN_conv, HGNN_fc
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter



class ComEDeFake(nn.Module):
    def __init__(self, in_n, in_e, n_out, n_class, n_hid, dropout=0.5, num_clusters=2, v=1):
        super(ComDeFake, self).__init__()
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.v = v

        self.hgn1 = HGNN_conv(in_n, n_hid)
        self.hgn2 = HGNN_conv(n_hid, n_out)

        self.hge1 = HGNN_conv(in_e, n_hid)
        self.hge2 = HGNN_conv(n_hid, n_out)

        self.cluster_layer = Parameter(torch.Tensor(num_clusters, n_out))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.fc = HGNN_fc(n_out, n_class)

    def forward(self, x, GV, y, GE):
        x = F.relu(self.hgn1(x, GV))
        x = F.dropout(x, self.dropout)
        x = self.hgn2(x, GV)

        y = F.relu(self.hge1(y, GE))
        y = F.dropout(y, self.dropout)
        y = self.hge2(y, GE)

        H_pred = self.decode(x, y)
        q = self.get_Q(x)
        x = self.fc(x)
        y = self.fc(y)

        return H_pred, x, q, y


    def decode(self, zn, ze):
        h = torch.mm(zn, ze.t())
        h = torch.sigmoid(h)
        return h


    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
