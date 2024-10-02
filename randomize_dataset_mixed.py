import random, os, shutil
import numpy as np

# Setup Destination
DATASET_INIRIAL_PATH = os.getcwd() + '/dataset_mixed/'
TRAIN_PATH = DATASET_INIRIAL_PATH + 'train/'
TEST_PATH = DATASET_INIRIAL_PATH + 'test/'

if not os.path.exists(DATASET_INIRIAL_PATH):
    os.mkdir(DATASET_INIRIAL_PATH)

if not os.path.exists(TRAIN_PATH):
    os.mkdir(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

# DATA_NUMBER = len(os.listdir(ORIGINAL_DEMO_PATH))
# TRAIN_NUMBER = int(DATA_NUMBER * 0.8)
# TEST_NUMBER = int(TRAIN_NUMBER + DATA_NUMBER * 0.1)

# Get Demo
ORIGINAL_DEMO_PATH = os.getcwd() + '/Pure_Rotation_Original_Demos/'
demo_folders = os.listdir(ORIGINAL_DEMO_PATH)

train_labels = []
train_index = 0
test_labels = []
test_index = 0
for folder in demo_folders:

    data_number = len(os.listdir(ORIGINAL_DEMO_PATH + folder)) - 1
    train_number = int(data_number * 0.9)

    labels = np.load(ORIGINAL_DEMO_PATH + folder + '/labels.npy')
    
    order = list(range(data_number))
    random.shuffle(order)

    for i in range(train_number):
        index = order[i]

        folder_name = ORIGINAL_DEMO_PATH + folder + '/' + format(index, '05d')
        new_folder_name = TRAIN_PATH + 'temp/' + format(train_index, '05d')
        shutil.copytree(folder_name, new_folder_name)   # Copy the file to train folder

        train_labels.append(labels[index])

        train_index += 1

    for i in range(train_number, data_number):
        index = order[i]

        folder_name = ORIGINAL_DEMO_PATH + folder + '/' + format(index, '05d')
        new_folder_name = TEST_PATH + 'temp/' + format(test_index, '05d')
        shutil.copytree(folder_name, new_folder_name)   # Copy the file to train folder

        test_labels.append(labels[index])

        test_index += 1

np.save((TRAIN_PATH + 'temp/' + 'labels'), train_labels)
np.save((TEST_PATH + 'temp/' + 'labels'), test_labels)


# Shuffle order inside train folder
data_number_train = len(os.listdir(TRAIN_PATH + 'temp/')) - 1
order = list(range(data_number_train))
random.shuffle(order)

labels = np.load(TRAIN_PATH + 'temp/' + 'labels.npy')
train_labels = []

for i in range(data_number_train):
    index = order[i]

    folder_name = TRAIN_PATH + 'temp/' + format(index, '05d')
    new_folder_name = TRAIN_PATH + format(i, '05d')
    os.rename(folder_name, new_folder_name)

    train_labels.append(labels[index])

np.save((TRAIN_PATH + 'labels'), train_labels)

# Shuffle order inside test folder
data_number_test = len(os.listdir(TEST_PATH + 'temp/')) - 1
order = list(range(data_number_test))
random.shuffle(order)

labels = np.load(TEST_PATH + 'temp/' + 'labels.npy')
test_labels = []

for i in range(data_number_test):
    index = order[i]

    folder_name = TEST_PATH + 'temp/' + format(index, '05d')
    new_folder_name = TEST_PATH + format(i, '05d')
    os.rename(folder_name, new_folder_name)

    test_labels.append(labels[index])

np.save((TEST_PATH + 'labels'), test_labels)

shutil.rmtree(TRAIN_PATH + 'temp/')
shutil.rmtree(TEST_PATH + 'temp/')