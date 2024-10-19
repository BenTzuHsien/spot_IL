import os
import numpy as np

DATASET_INITIAL_PATH = os.getcwd() + '/dataset_mixed/'
TRAIN_PATH = DATASET_INITIAL_PATH + 'train/'
TEST_PATH = DATASET_INITIAL_PATH + 'test/'

def quaternion_to_radians(quaternion):
    qx, qy, qz, qw = quaternion[-4:]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return yaw

def convert_labels_to_radians(path):
    labels_path = os.path.join(path, 'labels.npy')
    radians_labels_path = os.path.join(path, 'labels_radians.npy')
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        radians_labels = [quaternion_to_radians(label) for label in labels]
        np.save(radians_labels_path, radians_labels)
        print(f"Converted quaternions to radians and saved to {labels_path}")
      
convert_labels_to_radians(TRAIN_PATH)
convert_labels_to_radians(TEST_PATH)