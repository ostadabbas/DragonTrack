import cv2
import os
import numpy as np
import torch.nn.functional as F
from utils import *
from torch_geometric.data import Data, DataLoader, DataListLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
# def build_graph(tracklets, current_detections, images_path, current_frame, distance_limit, fps, positive_pairs, negative_pairs, test=True):
#     # Prepare next frame information
#     #next_frame_coords = []

#     # Function to find the detection in the next frame
# #     def find_next_frame_detection(detection, next_frame_detections):
# #         # Check if next_frame_detections is not used or is a placeholder
# #         if not next_frame_detections or next_frame_detections[0] is None:
# #             return [0, 0, 0, 0, 0, 0]  # Adjust the structure as needed

# #         for next_det in next_frame_detections:
# #             # Assuming index 1 is the ID; adjust according to your data structure
# #             if detection[1] == next_det[1]:
# #                 return next_det
# #         return None

#     if len(tracklets):
#         edges_first_row = []
#         edges_second_row = []
#         edges_complete_first_row= []
#         edges_complete_second_row = []
#         positional_edge_features = []
#         edge_attr = []
#         ground_truth = []
#         idx = []
#         node_attr = []
#         coords = []
#         frame = []
#         coords_original = []
#         transform= ToTensor()
#         ####tracklet graphs
#         for tracklet in tracklets:
#             tracklet1= tracklet[-1]
# #             next_det = find_next_frame_detection(tracklet1, next_frame_detections)
# #             if next_det is not None and len(next_det) > 0:
# #                 next_xmin, next_ymin, next_width, next_height = map(int, map(round, [next_det[2], next_det[3], next_det[4], next_det[5]]))
# #                 next_frame_coords.append([next_xmin, next_ymin, next_width, next_height])
# #             else:
# #                 next_frame_coords.append([0, 0, 0, 0])  # Or use a special token

#             xmin, ymin, width, height = int(round(tracklet1[2])), int(round(tracklet1[3])), \
#                                         int(round(tracklet1[4])), int(round(tracklet1[5]))
#             image_name = os.path.join(images_path, "{0:0=6d}".format(int(tracklet1[0])) + ".jpg")
#             image = plt.imread(image_name)
#             frame_width, frame_height, channels = image.shape
#             coords.append([xmin / frame_width, ymin / frame_height, width / frame_width, height / frame_height])
#             coords_original.append([xmin, ymin, xmin+width/2, ymin+height/2])
#             image_cropped = image[ymin:ymin + height, xmin:xmin + width]
#             image_resized = cv2.resize(image_cropped, (90,150), interpolation=cv2.INTER_AREA)
#             image_resized = image_resized / 255
#             image_resized = image_resized.astype(np.float32)
#             image_resized -= [0.485, 0.456, 0.406]
#             image_resized /= [0.229, 0.224, 0.225]
#             image_resized = transform(image_resized)
#             node_attr.append(image_resized)
#             frame.append([tracklet1[0]/fps])  # the frame it is observed
#         #####new detections graph
#         for detection in current_detections:
#             xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
#                                         int(round(detection[4])), int(round(detection[5]))
            
# #             next_det = find_next_frame_detection(detection, next_frame_detections)
# #             if next_det is not None and len(next_det) > 0:
# #                 next_xmin, next_ymin, next_width, next_height = map(int, map(round, [next_det[2], next_det[3], next_det[4], next_det[5]]))
# #                 next_frame_coords.append([next_xmin, next_ymin, next_width, next_height])
# #             else:
# #                 next_frame_coords.append([0, 0, 0, 0])  # Or use a special token
#             image_name = os.path.join(images_path, "{0:0=6d}".format(int(detection[0])) + ".jpg")
#             image = plt.imread(image_name)
#             frame_width, frame_height, channels = image.shape
#             coords.append([xmin / frame_width, ymin / frame_height, width / frame_width, height / frame_height])
#             coords_original.append([xmin, ymin, xmin+width/2, ymin+height/2])
#             image_cropped = image[ymin:ymin + height, xmin:xmin + width]
#             image_resized = cv2.resize(image_cropped, (90,150), interpolation=cv2.INTER_AREA)
#             image_resized = image_resized / 255
#             image_resized = image_resized.astype(np.float32)
#             image_resized -= [0.485, 0.456, 0.406]
#             image_resized /= [0.229, 0.224, 0.225]
#             image_resized = transform(image_resized)
#             node_attr.append(image_resized)
#             frame.append([detection[0]/fps])  # the frame it is observed
#         # construct connections between tracklets and detections
#         k = 0
#         for i in range(len(tracklets) + len(current_detections)):
#             for j in range(len(tracklets) + len(current_detections)):
#                 distance= ((coords_original[i][0]-coords_original[j][0])**2+(coords_original[i][1]-coords_original[j][1])**2)**0.5
#                 if i < len(tracklets) and j >= len(tracklets):  # i is tracklet j is detection
#                     # adjacency matrix
#                     if distance<distance_limit:
#                         edges_first_row.append(i)
#                         edges_second_row.append(j)
#                         edge_attr.append([0.0])
#                     if test==True:
#                         edges_complete_first_row.append(i)
#                         edges_complete_second_row.append(j)
#                     if tracklets[i][-1][1] == current_detections[j - len(tracklets)][1]:
#                         ground_truth.append(1.0)
#                     else:
#                         ground_truth.append(0.0)
#                     k += 1
#                 elif i >= len(tracklets) and j < len(tracklets):  # j is tracklet i is detection
#                     # adjacency matrix
#                     if distance<distance_limit:
#                         edges_first_row.append(i)
#                         edges_second_row.append(j)
#                         edge_attr.append([0.0])
#                     k += 1
        
#         # construct edge attributes for positional features
#         for i in range(len(edges_first_row)):
#             # Get the nodes connected by the edge
#             node_a_index = edges_first_row[i]
#             node_b_index = edges_second_row[i]

#             # Get their original coordinates (or normalized, depending on your preference)
#             node_a_coords = coords[node_a_index]
#             node_b_coords = coords[node_b_index]

#             # Calculate the difference in position
#             edge_feature = [node_a_coords[0] - node_b_coords[0],  # x difference
#                             node_a_coords[1] - node_b_coords[1],  # y difference
#                             node_a_coords[2] - node_b_coords[2],  # center x difference
#                             node_a_coords[3] - node_b_coords[3]]  # center y difference

#             positional_edge_features.append(edge_feature)
#         # Convert next frame coordinates to tensor
#         positional_edge_features = torch.tensor(positional_edge_features, dtype=torch.float)
#         # Handling positive and negative pairs
#         positive_pairs_tensor = torch.tensor(positive_pairs, dtype=torch.long)
#         negative_pairs_tensor = torch.tensor(negative_pairs, dtype=torch.long)
#         idx.append(current_frame - 2)
#         frame_node_attr = torch.stack(node_attr)
#         frame_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
#         frame_edges_index = torch.tensor([edges_first_row, edges_second_row], dtype=torch.long)
#         frame_coords = torch.tensor(coords, dtype=torch.float)
#         frame_ground_truth = torch.tensor(ground_truth, dtype=torch.float)
#         frame_idx = torch.tensor(idx, dtype=torch.float)
#         frame_edges_number = torch.tensor(len(edges_first_row), dtype=torch.int).reshape(1)
#         frame_frame = torch.tensor(frame, dtype=torch.float)
#         tracklets_frame = torch.tensor(len(tracklets), dtype=torch.float).reshape(1)
#         detections_frame = torch.tensor(len(current_detections), dtype=torch.float).reshape(1)
#         coords_original = torch.tensor(coords_original, dtype= torch.float)
#         edges_complete = torch.tensor([edges_complete_first_row, edges_complete_second_row], dtype=torch.long)
#         data = Data(x=frame_node_attr, edge_index=frame_edges_index, \
#                 edge_attr=frame_edge_attr, positional_edge_attr = positional_edge_features, coords=frame_coords, coords_original=coords_original,\
#                 ground_truth=frame_ground_truth, idx=frame_idx, \
#                 edges_number=frame_edges_number, frame=frame_frame, det_num=detections_frame, track_num=tracklets_frame, \
#                 edges_complete=edges_complete, positive_pairs=positive_pairs_tensor, negative_pairs=negative_pairs_tensor)
#         return data

def build_graph(tracklets, current_detections, images_path, current_frame, distance_limit, fps, test=True):
    # Prepare next frame information
    #next_frame_coords = []

    # Function to find the detection in the next frame
#     def find_next_frame_detection(detection, next_frame_detections):
#         # Check if next_frame_detections is not used or is a placeholder
#         if not next_frame_detections or next_frame_detections[0] is None:
#             return [0, 0, 0, 0, 0, 0]  # Adjust the structure as needed

#         for next_det in next_frame_detections:
#             # Assuming index 1 is the ID; adjust according to your data structure
#             if detection[1] == next_det[1]:
#                 return next_det
#         return None

    if len(tracklets):
        edges_first_row = []
        edges_second_row = []
        edges_complete_first_row= []
        edges_complete_second_row = []
        positional_edge_features = []
        edge_attr = []
        ground_truth = []
        idx = []
        node_attr = []
        coords = []
        frame = []
        coords_original = []
        transform= ToTensor()
        ####tracklet graphs
        for tracklet in tracklets:
            tracklet1= tracklet[-1]
#             next_det = find_next_frame_detection(tracklet1, next_frame_detections)
#             if next_det is not None and len(next_det) > 0:
#                 next_xmin, next_ymin, next_width, next_height = map(int, map(round, [next_det[2], next_det[3], next_det[4], next_det[5]]))
#                 next_frame_coords.append([next_xmin, next_ymin, next_width, next_height])
#             else:
#                 next_frame_coords.append([0, 0, 0, 0])  # Or use a special token

            xmin, ymin, width, height = int(round(tracklet1[2])), int(round(tracklet1[3])), \
                                        int(round(tracklet1[4])), int(round(tracklet1[5]))
            print(xmin, ymin, width, height)
            image_name = os.path.join(images_path, "{0:0=4d}".format(int(tracklet1[0])) + ".jpg")
            image = plt.imread(image_name)
            frame_width, frame_height, channels = image.shape
            coords.append([xmin / frame_width, ymin / frame_height, width / frame_width, height / frame_height])
            coords_original.append([xmin, ymin, xmin+width/2, ymin+height/2])
            image_cropped = image[ymin:ymin + height, xmin:xmin + width]
            image_resized = cv2.resize(image_cropped, (90,150), interpolation=cv2.INTER_AREA)
            image_resized = image_resized / 255
            image_resized = image_resized.astype(np.float32)
            image_resized -= [0.485, 0.456, 0.406]
            image_resized /= [0.229, 0.224, 0.225]
            image_resized = transform(image_resized)
            node_attr.append(image_resized)
            frame.append([tracklet1[0]/fps])  # the frame it is observed
        #####new detections graph
        for detection in current_detections:
            xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
                                        int(round(detection[4])), int(round(detection[5]))
            print(xmin, ymin, width, height)
#             next_det = find_next_frame_detection(detection, next_frame_detections)
#             if next_det is not None and len(next_det) > 0:
#                 next_xmin, next_ymin, next_width, next_height = map(int, map(round, [next_det[2], next_det[3], next_det[4], next_det[5]]))
#                 next_frame_coords.append([next_xmin, next_ymin, next_width, next_height])
#             else:
#                 next_frame_coords.append([0, 0, 0, 0])  # Or use a special token
            image_name = os.path.join(images_path, "{0:0=4d}".format(int(detection[0])) + ".jpg")
            image = plt.imread(image_name)
            frame_width, frame_height, channels = image.shape
            coords.append([xmin / frame_width, ymin / frame_height, width / frame_width, height / frame_height])
            coords_original.append([xmin, ymin, xmin+width/2, ymin+height/2])
            image_cropped = image[ymin:ymin + height, xmin:xmin + width]
            image_resized = cv2.resize(image_cropped, (90,150), interpolation=cv2.INTER_AREA)
            image_resized = image_resized / 255
            image_resized = image_resized.astype(np.float32)
            image_resized -= [0.485, 0.456, 0.406]
            image_resized /= [0.229, 0.224, 0.225]
            image_resized = transform(image_resized)
            node_attr.append(image_resized)
            frame.append([detection[0]/fps])  # the frame it is observed
        # construct connections between tracklets and detections
        k = 0
        for i in range(len(tracklets) + len(current_detections)):
            for j in range(len(tracklets) + len(current_detections)):
                distance= ((coords_original[i][0]-coords_original[j][0])**2+(coords_original[i][1]-coords_original[j][1])**2)**0.5
                if i < len(tracklets) and j >= len(tracklets):  # i is tracklet j is detection
                    # adjacency matrix
                    if distance<distance_limit:
                        edges_first_row.append(i)
                        edges_second_row.append(j)
                        edge_attr.append([0.0])
                    if test==True:
                        edges_complete_first_row.append(i)
                        edges_complete_second_row.append(j)
                    if tracklets[i][-1][1] == current_detections[j - len(tracklets)][1]:
                        ground_truth.append(1.0)
                    else:
                        ground_truth.append(0.0)
                    k += 1
                elif i >= len(tracklets) and j < len(tracklets):  # j is tracklet i is detection
                    # adjacency matrix
                    if distance<distance_limit:
                        edges_first_row.append(i)
                        edges_second_row.append(j)
                        edge_attr.append([0.0])
                    k += 1
        
        # construct edge attributes for positional features
        for i in range(len(edges_first_row)):
            # Get the nodes connected by the edge
            node_a_index = edges_first_row[i]
            node_b_index = edges_second_row[i]

            # Get their original coordinates (or normalized, depending on your preference)
            node_a_coords = coords[node_a_index]
            node_b_coords = coords[node_b_index]

            # Calculate the difference in position
            edge_feature = [node_a_coords[0] - node_b_coords[0],  # x difference
                            node_a_coords[1] - node_b_coords[1],  # y difference
                            node_a_coords[2] - node_b_coords[2],  # center x difference
                            node_a_coords[3] - node_b_coords[3]]  # center y difference

            positional_edge_features.append(edge_feature)
        # Convert next frame coordinates to tensor
        positional_edge_features = torch.tensor(positional_edge_features, dtype=torch.float)
        # Handling positive and negative pairs
#         positive_pairs_tensor = torch.tensor(positive_pairs, dtype=torch.long)
#         negative_pairs_tensor = torch.tensor(negative_pairs, dtype=torch.long)
        idx.append(current_frame - 2)
        frame_node_attr = torch.stack(node_attr)
        frame_edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        frame_edges_index = torch.tensor([edges_first_row, edges_second_row], dtype=torch.long)
        frame_coords = torch.tensor(coords, dtype=torch.float)
        frame_ground_truth = torch.tensor(ground_truth, dtype=torch.float)
        frame_idx = torch.tensor(idx, dtype=torch.float)
        frame_edges_number = torch.tensor(len(edges_first_row), dtype=torch.int).reshape(1)
        frame_frame = torch.tensor(frame, dtype=torch.float)
        tracklets_frame = torch.tensor(len(tracklets), dtype=torch.float).reshape(1)
        detections_frame = torch.tensor(len(current_detections), dtype=torch.float).reshape(1)
        coords_original = torch.tensor(coords_original, dtype= torch.float)
        edges_complete = torch.tensor([edges_complete_first_row, edges_complete_second_row], dtype=torch.long)
        data = Data(x=frame_node_attr, edge_index=frame_edges_index, \
                edge_attr=frame_edge_attr, positional_edge_attr = positional_edge_features, coords=frame_coords, coords_original=coords_original,\
                ground_truth=frame_ground_truth, idx=frame_idx, \
                edges_number=frame_edges_number, frame=frame_frame, det_num=detections_frame, track_num=tracklets_frame, \
                edges_complete=edges_complete)
        return data