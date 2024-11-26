import motmetrics as mm
import pandas as pd
import numpy as np

def parse_tracking_data_to_df(file_path):
    """
    Parses tracking data from a given file path and converts it to a DataFrame.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            frame_number, object_id, x, y, width, height = line.strip().split(',')[:6]
            x, y, width, height = map(float, [x, y, width, height])
            data.append([int(frame_number), int(object_id), x, y, x + width, y + height])
    return pd.DataFrame(data, columns=['Frame', 'ID', 'X', 'Y', 'X2', 'Y2'])

def main():
    acc = mm.MOTAccumulator(auto_id=True)
    mh = mm.metrics.create()

    sequences = ['02']
    results = []

    for seq in sequences:
        gt_file = f'/home/Galoaa.b/ondemand/data/MOT17/train/MOT17-{seq}-FRCNN/gt/gt.txt'
        pred_file = f'/home/galoaa.b/ondemand/dev/GCNNMatch/output_v2/MOT17-{seq}-FRCNN.txt'
        
        gt_df = parse_tracking_data_to_df(gt_file)
        pred_df = parse_tracking_data_to_df(pred_file)

        for frame in sorted(gt_df['Frame'].unique()):
            gt_frame_data = gt_df[gt_df['Frame'] == frame]
            pred_frame_data = pred_df[pred_df['Frame'] == frame]

            distances = mm.distances.iou_matrix(gt_frame_data[['X', 'Y', 'X2', 'Y2']], pred_frame_data[['X', 'Y', 'X2', 'Y2']], max_iou=0.5)

            acc.update(
                gt_frame_data['ID'].values,
                pred_frame_data['ID'].values,
                distances
            )

        summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches'], name='acc')
        print(f'Sequence {seq}:', summary)

        results.append(summary)
        
if __name__ == "__main__":
    main()