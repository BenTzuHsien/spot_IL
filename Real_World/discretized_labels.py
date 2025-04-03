import os
import numpy as np

DATASET_DIR = '/data/lee04484/SPOT_Real_World_Dataset/cleanup_dataset'
maps = [x for x in os.listdir(DATASET_DIR) if 'map' in x]

def convert_label(labels):
    new_labels = []
    
    for label in labels:
        new_label = [1, 1, 1]

        for i in range(3):
            if label[i] > 0.01:
                new_label[i] = 2
            elif label[i] < -0.01:
                new_label[i] = 0
        new_labels.append(new_label)
    
    return np.array(new_labels)

for map in maps:
    map_path = os.path.join(DATASET_DIR, map)
    num_traj = len(os.listdir(map_path)) - 1

    for i in range(num_traj):
        traj = f'traj_{i:03}'
        labels_path = os.path.join(map_path, traj, 'labels.npy')

        labels = np.load(labels_path)
        new_labels = convert_label(labels)

        new_labels_path = os.path.join(map_path, traj, 'discretized_labels.npy')
        np.save(new_labels_path, new_labels)