import sys
import torch.nn as nn
import os
import numpy as np
from network.optimizationGNN import optimNet
from network.encoderCNN import EncoderCNN
from network.detections import *
from network.affinity import affinityNet
from network.affinity_appearance import affinity_appearanceNet
from network.affinity_geom import affinity_geomNet
from network.positional_affinity_net import PositionalAffinityNet
from network.learnable_sinkhorn import LearnableSinkhorn
from torch.nn.parameter import Parameter
from utils import *
import torch
# from torch_sparse import transpose
from network.affinity_final import *


def compute_contrastive_loss(pos_embeddings_a, pos_embeddings_b, neg_embeddings_a, neg_embeddings_b, margin=1.0):
    # Calculate the Euclidean distance for positive and negative pairs
    positive_distance = torch.nn.functional.pairwise_distance(pos_embeddings_a, pos_embeddings_b, p=2)
    negative_distance = torch.nn.functional.pairwise_distance(neg_embeddings_a, neg_embeddings_b, p=2)
    
    # Contrastive loss formula
    loss_pos = torch.mean(torch.pow(positive_distance, 2))
    loss_neg = torch.mean(torch.pow(torch.clamp(margin - negative_distance, min=0.0), 2))
    contrastive_loss = loss_pos + loss_neg
    return contrastive_loss


class completeNet(nn.Module):
    def __init__(self):
        super(completeNet, self).__init__()

        self.cnn = EncoderCNN()
        self.affinity_net = affinityNet()
        self.affinity_appearance_net= affinity_appearanceNet()
        self.affinity_geom_net= affinity_geomNet()
        self.affinity_final_net= affinity_finalNet()
        self.positional_affinity_net = PositionalAffinityNet()
        # self.learnable_sinkhorn = LearnableSinkhorn(max_iter=10)
        self.optim_net = optimNet()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
#         self.slack = nn.Parameter(torch.tensor([-0.2], dtype=torch.float32), requires_grad=True)
#         self.lam = nn.Parameter(torch.tensor([5.0], dtype=torch.float32), requires_grad=True)


    def forward(self, data):
        # print(data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, coords_original, edge_index, ground_truth, coords, edges_number_list, frame, track_num, detections_num, positional_edge_attr = \
            data.x, data.coords_original, data.edge_index, data.ground_truth, data.coords, data.edges_number, data.frame, data.track_num, data.det_num, data.positional_edge_attr
        slack= torch.Tensor([-0.2]).float().cuda()
        lam= torch.Tensor([5]).float().cuda()
        #Pass through GNN
        node_embedding= self.cnn(x)
        # print("node_embedding size after CNN:", node_embedding.size())
        edge_embedding = []
        edge_mlp= []
        for i in range(len(edge_index[0])):
            #CNN features
            x1 = self.affinity_appearance_net(torch.cat((node_embedding[edge_index[0][i]], node_embedding[edge_index[1][i]]), 0))
            #geometry
            x2 = self.affinity_geom_net(torch.cat((coords[edge_index[0][i]], coords[edge_index[1][i]]), 0))
            #iou
            iou= box_iou_calc(coords_original[edge_index[0][i]], coords_original[edge_index[1][i]])
            # x2= iou
            edge_mlp.append(iou)
            #pass through mlp
            inputs = torch.cat((x1.reshape(1), x2.reshape(1)), 0)
            edge_embedding.append(self.affinity_net(inputs))
    
        # print(edge_embedding)
        # print("Edge embedding size before stacking:", [e.shape for e in edge_embedding])
        edge_embedding = torch.stack(edge_embedding)
        # print("Edge embedding size after stacking:", edge_embedding.shape)


        processed_positional_features = []
        for pos_attr in positional_edge_attr:
            processed_feature = self.positional_affinity_net(pos_attr.unsqueeze(0))
            processed_positional_features.append(processed_feature)

        # print("Positional edge attributes size before cat:", [p.shape for p in processed_positional_features])
        processed_positional_features = torch.cat(processed_positional_features, dim=0)
        # print("processed_positional_features size:", processed_positional_features.size())  # Add this line

        # print("Processed positional features size after cat:", processed_positional_features.shape)

        combined_edge_features = torch.cat([edge_embedding, processed_positional_features], dim=1)
        # print("Combined edge features size:", combined_edge_features.shape)
        # print("node_embedding size:", node_embedding.size())  # Add this line
        # print("combined_edge_features size:", combined_edge_features.size())  # Add this line
        output = self.optim_net(node_embedding, combined_edge_features, edge_index, coords, frame)
        # Extract embeddings for positive and negative pairs based on indices
#         pos_embeddings_a = output[data.positive_pairs[:, 0]]
#         pos_embeddings_b = output[data.positive_pairs[:, 1]]
#         neg_embeddings_a = output[data.negative_pairs[:, 0]]
#         neg_embeddings_b = output[data.negative_pairs[:, 1]]
        
#         # Compute contrastive loss
#         contrastive_loss = compute_contrastive_loss(pos_embeddings_a, pos_embeddings_b, neg_embeddings_a, neg_embeddings_b)
        
        output_temp= []
        for i in range(len(edge_index[0])):
            if edge_index[0][i]<edge_index[1][i]:
                nodes_difference= self.cos(output[edge_index[0][i]], output[edge_index[1][i]])
                x1 = self.affinity_final_net(torch.cat((nodes_difference.reshape(1), edge_mlp[i].reshape(1)), 0))
                output_temp.append(x1.reshape(1))
        output= output_temp
        start1= 0
        start2 = 0 #two are used here because output is already reduced while edges not
        normalized_output= []
        tracklet_num = []
        det_num = []
        for i,j in enumerate(data.idx):
            num_of_edges1= edges_number_list[i].item()
            num_of_edges2= int(num_of_edges1/2)
            output_sliced= output[start2:start2+num_of_edges2]
            edges_sliced= edge_index[:, start1:start1+num_of_edges1]
            start1 += num_of_edges1
            start2 += num_of_edges2

            row, col = edges_sliced
            mask = row < col
            edges_sliced = edges_sliced[:, mask]
            num_of_nodes= sum(track_num[0:i])+sum(detections_num[0:i])
            for k,l  in enumerate(edges_sliced):
                for m,n in enumerate(l): 
                    edges_sliced[k,m]= edges_sliced[k,m]-num_of_nodes
            # Construct matrix for Sinkhorn normalization
            # elevate to e power and augment with slack variable

            matrix = []
            for k in range(int(track_num[i].item())):
                matrix.append([])
                for l in range(int(detections_num[i].item())):
                    matrix[k].append(torch.zeros(1, dtype=torch.float, requires_grad=False).to(device))
                matrix[k].append(torch.exp(slack*lam))#slack
            for k,m in enumerate(edges_sliced[0]):
                matrix[int(edges_sliced[0,k].item())][int(edges_sliced[1,k].item())-int(track_num[i].item())]= torch.exp(output_sliced[k]*lam)
            for w,z in enumerate(matrix):
                matrix[w] = torch.cat(z)
            matrix.append(torch.ones(len(matrix[0])).to(device)*torch.exp(slack*lam))#slack
            matrix = torch.stack(matrix)
            matrix = sinkhorn(matrix)
            matrix = matrix[0:-1,0:-1]
            det_num.append(torch.tensor(len(matrix[0]), dtype= int).to(device))
            tracklet_num.append(torch.tensor(len(matrix), dtype= int).to(device))
            normalized_output.append(matrix.reshape(-1))
        normalized_output = torch.cat((normalized_output[:]),dim=0)
        normalized_output_final= []
        ground_truth_final= []
        for k, l in enumerate(normalized_output):
            if l.item()!=0:
                normalized_output_final.append(l)
                ground_truth_final.append(ground_truth[k])
#         print("Stacked Normalized Output:", torch.stack(normalized_output_final))
#         print("Normalized Outout:", normalized_output)
#         print("Stacked GT:", torch.stack(ground_truth_final))
#         print("Ground Truth:", ground_truth)
#         print("Stack Detection Number:", torch.stack(det_num)) 
#         print("Stacked Tracklet Number:", torch.stack(tracklet_num))
#         print('-----------------------------------------')
        return torch.stack(normalized_output_final), normalized_output, torch.stack(ground_truth_final), ground_truth, torch.stack(det_num), torch.stack(tracklet_num)#, contrastive_loss

        #     matrix = []
        #     for k in range(int(track_num[i].item())):
        #         matrix_row = [torch.exp(output_sliced[l] * lam) if l < num_of_edges2 else torch.zeros(1, dtype=torch.float, requires_grad=False).to(device) 
        #                       for l in range(int(detections_num[i].item()))]
        #         matrix_row.append(torch.exp(slack * lam))  # slack
        #         matrix.append(torch.cat(matrix_row))

        #     # Add a row for slack in matrix
        #     slack_row = torch.ones(len(matrix[0])).to(device) * torch.exp(slack * lam)  # slack
        #     matrix.append(slack_row)
        #     matrix = torch.stack(matrix)

        #     # Apply learnable Sinkhorn normalization
        #     matrix = self.learnable_sinkhorn(matrix)
        #     matrix = matrix[:-1, :-1]  # Remove slack row and column

        #     det_num.append(torch.tensor(len(matrix[0]), dtype=int).to(device))
        #     tracklet_num.append(torch.tensor(len(matrix), dtype=int).to(device))
        #     normalized_output.append(matrix.reshape(-1))

        # normalized_output = torch.cat(normalized_output, dim=0)
        # normalized_output_final = []
        # ground_truth_final = []
        # for k, l in enumerate(normalized_output):
        #     if l.item() != 0:
        #         normalized_output_final.append(l)
        #         ground_truth_final.append(ground_truth[k])
        # print("Normalized Output:", normalized_output)
        # return torch.stack(normalized_output_final), normalized_output, torch.stack(ground_truth_final), ground_truth, torch.stack(det_num), torch.stack(tracklet_num)


        #     # elevate to e power and augment with slack variable
        #     matrix = []
        #     for k in range(int(track_num[i].item())):
        #         matrix.append([])
        #         for l in range(int(detections_num[i].item())):
        #             matrix[k].append(torch.zeros(1, dtype=torch.float, requires_grad=False).to(device))
        #         matrix[k].append(torch.exp(slack*lam))#slack
        #     for k,m in enumerate(edges_sliced[0]):
        #         matrix[int(edges_sliced[0,k].item())][int(edges_sliced[1,k].item())-int(track_num[i].item())]= torch.exp(output_sliced[k]*lam)
        #     for w,z in enumerate(matrix):
        #         matrix[w] = torch.cat(z)
        #     matrix.append(torch.ones(len(matrix[0])).to(device)*torch.exp(slack*lam))#slack
        #     matrix = torch.stack(matrix)
        #     matrix = sinkhorn(matrix)
        #     matrix = matrix[0:-1,0:-1]
        #     det_num.append(torch.tensor(len(matrix[0]), dtype= int).to(device))
        #     tracklet_num.append(torch.tensor(len(matrix), dtype= int).to(device))
        #     normalized_output.append(matrix.reshape(-1))
        # normalized_output = torch.cat((normalized_output[:]),dim=0)
        # normalized_output_final= []
        # ground_truth_final= []
        # for k, l in enumerate(normalized_output):
        #     if l.item()!=0:
        #         normalized_output_final.append(l)
        #         ground_truth_final.append(ground_truth[k])
        # return torch.stack(normalized_output_final), normalized_output, torch.stack(ground_truth_final), ground_truth, torch.stack(det_num), torch.stack(tracklet_num)