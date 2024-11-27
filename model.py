import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn.conv import GCNConv


def load_model(name, input_dim, hid_dim, output_dim, dropout):
    if name == "gcn":
        return GCN(input_dim, hid_dim, output_dim, dropout)
    elif name == "gat":
        return GAT(input_dim, hid_dim, output_dim, dropout=dropout)
    
class GCN(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim, p=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, output_dim)
        self.p = p
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        embedding = F.dropout(z, p=self.p)
        logits = self.conv2(embedding, edge_index)
        return logits, embedding, None
    
    
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(GATConv(input_dim, hid_dim)) 
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hid_dim, hid_dim))
            self.layers.append(GATConv(hid_dim, output_dim))
        else:
            self.layers.append(GATConv(input_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        esoft_list = []
        
        for i, layer in enumerate(self.layers[:-1]):
            x, (edge_index, alpha) = layer(x, edge_index, return_attention_weights=True)
            esoft_list.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits, (edge_index, alpha) = self.layers[-1](x, edge_index, return_attention_weights=True)
        esoft_list.append(alpha) 

        embedding = x
        return logits, embedding, esoft_list



    def decode(self, embedding, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        logits = cos(embedding[edge_index[0]], embedding[edge_index[1]])
        logits = (logits+1)/2
        return logits
