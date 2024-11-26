from network.encoderCNN import *
import sys
import torch.nn as nn
import os
import numpy as np
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch



class optimNet(nn.Module):
    def __init__(self):
        super(optimNet, self).__init__()
        # Assuming each node feature is 1024-dimensional
        self.node_feature_size = 1024
        # After transformation
        self.edge_feature_size = 1
        # GCNConv layers with adjusted input sizes
        self.conv1 = GCNConv(2049, 512, improved=False, cached=False, bias=True)
        self.conv2 = GCNConv(1408, 128, improved=False, cached=False, bias=True)
        # Linear layer to transform 2D edge features to 1D
        self.edge_feature_transform = nn.Linear(2, self.edge_feature_size)
        # Separate MLPs for node and positional features
        self.node_feature_mlp = nn.Sequential(
            nn.Linear(512, 128),  # Adjust input size here
            nn.ReLU()
        )
        self.edge_feature_mlp = nn.Sequential(
            nn.Linear(self.edge_feature_size, 128),
            nn.ReLU()
        )

    def similarity1(self, node_embedding, edge_index, edge_attr_transformed):
        edge_attr = []
        for i in range(len(edge_index[0])):
            # Use separate MLPs for node features and edge features
            node_feat_src = self.node_feature_mlp(node_embedding[edge_index[0][i]].unsqueeze(0))
            node_feat_dst = self.node_feature_mlp(node_embedding[edge_index[1][i]].unsqueeze(0))
            edge_feat = self.edge_feature_mlp(edge_attr_transformed[i].unsqueeze(0))

            # Concatenate features along dimension 1
            concatenated_features = torch.cat((node_feat_src, node_feat_dst, edge_feat), dim=1)
            x1 = F.relu(concatenated_features)
            edge_attr.append(x1.squeeze(0))
        edge_attr = torch.stack(edge_attr)
        return edge_attr

    def forward(self, node_attr, edge_attr, edge_index, coords, frame):
        node_embedding = node_attr

        # Transform edge features once
        edge_attr_transformed = self.edge_feature_transform(edge_attr)

        # Concatenate node features with transformed edge features for each edge
        src = node_embedding[edge_index[0]]  # Source node features
        dst = node_embedding[edge_index[1]]  # Destination node features
        edge_input = torch.cat((src, dst, edge_attr_transformed), dim=1)

        # First GCNConv layer
        out = self.conv1(edge_input, edge_index)
        out = F.relu(out)

        # Calculate similarity with separate MLPs
        edge_attr = self.similarity1(out, edge_index, edge_attr_transformed)

        # Concatenate for the second GCNConv layer
        edge_input = torch.cat((out[edge_index[0]], out[edge_index[1]], edge_attr), dim=1)

        # Second GCNConv layer
        out = self.conv2(edge_input, edge_index)

        return out

## Attention Based Model ##
# class optimNet(nn.Module):
#     def __init__(self):
#         super(optimNet, self).__init__()
#         self.node_feature_size = 1024
#         self.edge_feature_size = 1

#         self.conv1 = GCNConv(2049, 512)
#         self.conv2 = GCNConv(1408, 128)
#         self.edge_feature_transform = nn.Linear(2, self.edge_feature_size)

#         self.node_feature_mlp = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU()
#         )
#         self.edge_feature_mlp = nn.Sequential(
#             nn.Linear(self.edge_feature_size, 128),
#             nn.ReLU()
#         )

#         self.attention = nn.MultiheadAttention(128, 2)

#     def similarity1(self, node_embedding, edge_index, edge_attr_transformed, coords):
#         edge_attr = []
#         for i in range(len(edge_index[0])):
#             node_feat_src = self.node_feature_mlp(node_embedding[edge_index[0][i]].unsqueeze(0))
#             node_feat_dst = self.node_feature_mlp(node_embedding[edge_index[1][i]].unsqueeze(0))
#             edge_feat = self.edge_feature_mlp(edge_attr_transformed[i].unsqueeze(0))

#             concatenated_features = torch.cat((node_feat_src, node_feat_dst, edge_feat), dim=1)
#             x1 = F.relu(concatenated_features)

#             attention_scores, _ = self.attention(edge_feat, coords[i].unsqueeze(0))
#             attention_weights = torch.softmax(attention_scores, dim=1)
#             weighted_features = attention_weights[0, 0] * edge_feat + attention_weights[0, 1] * coords[i]

#             concatenated_features = torch.cat((concatenated_features, weighted_features), dim=1)
#             x1 = F.relu(concatenated_features)
#             edge_attr.append(x1.squeeze(0))

#         edge_attr = torch.stack(edge_attr)
#         return edge_attr

#     def forward(self, node_attr, edge_attr, edge_index, coords):
#         node_embedding = node_attr

#         edge_attr_transformed = self.edge_feature_transform(edge_attr)

#         src = node_embedding[edge_index[0]]
#         dst = node_embedding[edge_index[1]]
#         edge_input = torch.cat((src, dst, edge_attr_transformed), dim=1)

#         out = self.conv1(edge_input, edge_index)
#         out = F.relu(out)

#         edge_attr = self.similarity1(out, edge_index, edge_attr_transformed, coords)

#         edge_input = torch.cat((out[edge_index[0]], out[edge_index[1]], edge_attr), dim=1)

#         out = self.conv2(edge_input, edge_index)

#         return out


# class optimNet(nn.Module):
#     def __init__(self):
#             super(optimNet, self).__init__()
#             # Assuming each node feature is 1024-dimensional
#             self.node_feature_size = 1024
#             self.edge_feature_size = 1  # After transformation

#             # GCNConv layers with adjusted input sizes
#             # Adjust the input feature size to match the concatenated node and edge features
#             self.conv1 = GCNConv(2049, 512, improved=False, cached=False, bias=True)
#             self.conv2 = GCNConv(513, 128, improved=False, cached=False, bias=True)

#             # MLP for node feature concatenation
#             self.mlp1 = nn.Sequential(
#                 nn.Linear(1024, 1),
#                 nn.ReLU()
#             )

#             # Linear layer to transform 2D edge features to 1D
#             self.edge_feature_transform = nn.Linear(2, self.edge_feature_size)


#     def similarity1(self, node_embedding, edge_index):
#         edge_attr = []
#         for i in range(len(edge_index[0])):
#             # Print the indices being used for debugging
#             # print("edge_index[0][i]:", edge_index[0][i].item(), "edge_index[1][i]:", edge_index[1][i].item())

#             node_feat_src = node_embedding[edge_index[0][i]].unsqueeze(0)
#             node_feat_dst = node_embedding[edge_index[1][i]].unsqueeze(0)

#             # Print sizes of individual features for debugging
#             # print("node_feat_src size:", node_feat_src.size(), "node_feat_dst size:", node_feat_dst.size())

#             concatenated_features = torch.cat((node_feat_src, node_feat_dst), dim=1)

#             # Debugging print statement
#             # print("Concatenated features size after adjustment:", concatenated_features.size())

#             x1 = self.mlp1(concatenated_features)
#             edge_attr.append(x1.squeeze(0))
#         edge_attr = torch.stack(edge_attr)
#         print("edge_attr size after similarity1:", edge_attr.size())
#         return edge_attr
    
#     def forward(self, node_attr, edge_attr, edge_index, coords, frame):
#         node_embedding = node_attr

#         # Transform edge features once
#         edge_attr_transformed = self.edge_feature_transform(edge_attr)

#         # Concatenate node features with transformed edge features for each edge
#         src = node_embedding[edge_index[0]]  # Source node features
#         dst = node_embedding[edge_index[1]]  # Destination node features
#         edge_input = torch.cat((src, dst, edge_attr_transformed), dim=1)

#         # First GCNConv layer
#         out = self.conv1(edge_input, edge_index)
#         out = F.relu(out)

#         # Calculate similarity
#         edge_attr = self.similarity1(out, edge_index)

#         # Concatenate for the second GCNConv layer
#         edge_input = torch.cat((out[edge_index[0]], out[edge_index[1]], edge_attr), dim=1)

#         # Second GCNConv layer
#         out = self.conv2(edge_input, edge_index)

#         return out
