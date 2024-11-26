import motmetrics as mm
import numpy as np

def calculate_assa(correct_id_matches, tp):
    return correct_id_matches / tp if tp > 0 else 0

def calculate_deta(tp, fp, fn):
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

def calculate_id_metrics(frame_gt_data, frame_pred_data, matches):
    """
    Calculate True Positives (TP), False Positives (FP), False Negatives (FN),
    necessary for calculating IDF1.
    """
    TP = len(matches)
    FP = len(frame_pred_data) - TP
    FN = len(frame_gt_data) - TP
    return TP, FP, FN

def update_id_switches(prev_matches, current_matches):
    """
    Calculate ID Switches (IDS) based on changes in matched IDs from the previous frame.
    """
    IDS = sum(1 for gt_id, pred_id in current_matches.items() if gt_id in prev_matches and prev_matches[gt_id] != pred_id)
    return IDS

def parse_tracking_data(file_path):
    """
    Parses tracking data from a given file path.
    Returns a dictionary with frame numbers as keys and a list of object
    detections as values.
    """
    tracking_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_number, object_id, x, y, width, height = parts[:6]
            frame_number = int(frame_number)
            detection = {'object_id': int(object_id), 'bbox': (float(x), float(y), float(width), float(height))}
            if frame_number not in tracking_data:
                tracking_data[frame_number] = []
            tracking_data[frame_number].append(detection)
    return tracking_data


def calculate_correct_id_matches(frame_gt_data, frame_pred_data, matches):
    correct_id_matches = 0
    for gt_index, pred_index in matches:
        gt_id = frame_gt_data[gt_index]['object_id']
        pred_id = frame_pred_data[pred_index]['object_id']
        if gt_id == pred_id:
            correct_id_matches += 1
        else:
            print(f"Mismatch: GT ID {gt_id} vs Pred ID {pred_id}")  # Debug print
    return correct_id_matches

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def match_detections(frame_gt_data, frame_pred_data, iou_threshold=0.8):
    """
    Match detections between ground truth and predictions based on IoU.
    Returns a list of matches (gt_index, pred_index).
    """
    matches = []
    used_predictions = set()
    partial_matches = []  # For partial credit

    for gt_index, gt_det in enumerate(frame_gt_data):
        best_iou = iou_threshold
        best_pred_index = -1
        for pred_index, pred_det in enumerate(frame_pred_data):
            if pred_index in used_predictions:
                continue
            iou = calculate_iou(gt_det['bbox'], pred_det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_pred_index = pred_index
            elif iou_threshold > iou > iou_threshold * 0.75:  # Example condition for partial credit
                partial_matches.append((gt_index, pred_index, iou))  # Track partial matches

        if best_pred_index != -1:
            matches.append((gt_index, best_pred_index))
            used_predictions.add(best_pred_index)

    return matches


def match_detections_with_ids(frame_gt_data, frame_pred_data, prev_matches, iou_threshold=0.8):
    """
    Match detections between ground truth and predictions based on IoU, considering ID switches.
    `prev_matches` is a dictionary mapping previous frame's ground truth IDs to predicted IDs to track continuity.
    """
    matches = []
    id_switches = 0
    used_predictions = set()
    current_matches = {}

    for gt_index, gt_det in enumerate(frame_gt_data):
        best_iou = iou_threshold
        best_pred_index = -1
        for pred_index, pred_det in enumerate(frame_pred_data):
            if pred_index in used_predictions:
                continue
            iou = calculate_iou(gt_det['bbox'], pred_det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_pred_index = pred_index
            elif iou_threshold > iou > iou_threshold * 0.75:  # Example condition for partial credit
                partial_matches.append((gt_index, pred_index, iou))  # Track partial matches

        if best_pred_index != -1:
            gt_id = gt_det['object_id']
            pred_id = frame_pred_data[best_pred_index]['object_id']
            current_matches[gt_id] = pred_id
            used_predictions.add(best_pred_index)
            
            # Check for ID switch
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                id_switches += 1

            matches.append((gt_index, best_pred_index))

    return matches, id_switches, current_matches

def calculate_mota(ground_truth_data, predicted_data, iou_threshold=0.8):
    total_misses = 0
    total_false_positives = 0
    total_id_switches = 0
    total_objects = sum(len(detections) for detections in ground_truth_data.values())
    prev_matches = {}  # Track matches from the previous frame for ID switch detection

    for frame in sorted(set(ground_truth_data.keys()).union(predicted_data.keys())):
        frame_gt_data = ground_truth_data.get(frame, [])
        frame_pred_data = predicted_data.get(frame, [])
        
        matches, id_switches, current_matches = match_detections_with_ids(frame_gt_data, frame_pred_data, prev_matches, iou_threshold=iou_threshold)
        prev_matches = current_matches  # Update matches for the next frame
        
        match_indices = set(index for _, index in matches)
        total_misses += len(frame_gt_data) - len(matches)
        total_false_positives += len(frame_pred_data) - len(match_indices)
        total_id_switches += id_switches

    mota = 1 - (total_misses + total_false_positives + (0.15*total_id_switches)) / total_objects if total_objects > 0 else 0
    return mota

def calculate_tracking_metrics(ground_truth_data, predicted_data, iou_threshold=0.8):
    total_TP = total_FP = total_FN = total_IDS = 0  # ID switches are not applicable here
    correct_matches_per_frame = []  # Collect match counts per frame for an adapted "AssA"

    for frame in sorted(set(ground_truth_data.keys()).union(predicted_data.keys())):
        frame_gt_data = ground_truth_data.get(frame, [])
        frame_pred_data = predicted_data.get(frame, [])
        matches = match_detections(frame_gt_data, frame_pred_data, iou_threshold=iou_threshold)
        
        TP, FP, FN = calculate_id_metrics(frame_gt_data, frame_pred_data, matches)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        correct_matches_per_frame.append(len(matches))  # Using match count as a stand-in for AssA

    # Traditional metrics
    id_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    id_recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    idf1 = 2 * (id_precision * id_recall) / (id_precision + id_recall) if id_precision + id_recall > 0 else 0
    deta = calculate_deta(total_TP, total_FP, total_FN)
    
    # Adapted AssA calculation
    adapted_assa = sum(correct_matches_per_frame) / total_TP if total_TP > 0 else 0
    hota = (deta + adapted_assa) / 2  # Note: This HOTA is adapted and not standard

    return hota, deta, adapted_assa, idf1, total_IDS

def main():
    sequences = [1, 3, 6, 8]  # Define the sequences to process
    overall_hota, overall_deta, overall_assa, overall_mota, overall_idf1, overall_ids = 0, 0, 0, 0, 0, 0

    for seq in sequences:
        gt_file = f'/home/Galoaa.b/ondemand/data/MOT17/test/MOT17-0{seq}-FRCNN/det/det.txt'
        pred_file = f'/home/galoaa.b/ondemand/dev/GCNNMatch/output_v2/MOT17-0{seq}-FRCNN.txt'

        ground_truth_data = parse_tracking_data(gt_file)
        predicted_data = parse_tracking_data(pred_file)
        mota = calculate_mota(ground_truth_data, predicted_data)
        # Calculate the metrics for this sequence
        hota, deta, assa, idf1, ids = calculate_tracking_metrics(ground_truth_data, predicted_data)

        # Accumulate the metrics
        overall_hota += hota
        overall_deta += deta
        overall_assa += assa
        overall_mota += mota
        overall_idf1 += idf1
        overall_ids += ids

        # Optionally print metrics for each sequence
        print(f"Sequence MOT17-0{seq} Metrics: HOTA: {hota:.4f}, DetA: {deta:.4f}, AssA: {assa:.4f}, MOTA:{mota:4f} ,IDF1: {idf1:.4f}, IDS: {ids}")

    # Average the metrics over all sequences
    num_sequences = len(sequences)
    avg_hota = overall_hota / num_sequences
    avg_deta = overall_deta / num_sequences
    avg_assa = overall_assa / num_sequences
    avg_mota = overall_mota / num_sequences
    avg_idf1 = overall_idf1 / num_sequences
    avg_ids = overall_ids
    # Print overall metrics
    print(f"Overall Metrics Across Sequences: HOTA: {avg_hota:.4f}, DetA: {avg_deta:.4f}, AssA: {avg_assa:.4f},MOTA: {avg_mota:.4f},  IDF1: {avg_idf1:.4f}, IDS: {avg_ids}")

if __name__ == "__main__":
    main()