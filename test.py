from torch.utils.data import DataLoader
import torch, os
from torchvision import transforms
from spotdatasetloader import SPOTDataLoader
from FiveResNet18MLP5_7 import FiveResNet18MLP5_7
import numpy as np
import matplotlib.pyplot as plt

# Parameters
WEIGHT_DATAET_NAME = 'mixed'
TEST_DATASET_NAME = 'mixed'
TEST_DATASET_TYPE = 'test'
FOLD = 0
WEIGHT_NAME = 'epoch_1718.pth'

# Constant
DATASET_PATH = os.getcwd() + f'/dataset_{TEST_DATASET_NAME}/'
TEST_PATH = DATASET_PATH + f'{TEST_DATASET_TYPE}/'
GOAL_PATH = DATASET_PATH + 'goal/'
LABEL_PATH = TEST_PATH + 'labels.npy'
WEIGHT_PATH = os.getcwd() + f'/weights/FiveResNet18MLP5_{WEIGHT_DATAET_NAME}/lr1e-6_full_output/fold_{FOLD}/'
FIGURES_PATH = os.getcwd() + f'/Results/FiveResNet18MLP5_{WEIGHT_DATAET_NAME}/lr1e-6_full_output/test/fold_{FOLD}/'
if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

DPI = 120
FIGURE_SIZE_PIXEL = [2490, 1490]
FIGURE_SIZE = [fsp / DPI for fsp in FIGURE_SIZE_PIXEL]

def test_model(test_dataset, model, weight_name, device='cuda', draw=False, show=False):
    
    model.to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH + weight_name))
    model.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=1)
    results = np.empty([0, 14])

    with torch.no_grad():
        idx = 0
        for current_images, goal_images, label in test_dataloader:
            output = model(current_images, goal_images)
            # output_degree = (output.item() / np.pi) * 180
            # label_degree = (label.item() / np.pi) * 180
            # loss = abs(label - output)

            iteration_result = np.array([output.flatten().cpu().numpy(), label.flatten().cpu().numpy()])
            results = np.vstack([results, iteration_result.flatten()])

            print(idx, iteration_result)
            idx += 1

    # Plot
    if draw is True:
        plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        plt.plot(range(len(test_dataloader)), results[:, 0], color='green', linestyle='-', label='Predicted Rotation Angle')
        plt.plot(range(len(test_dataloader)), results[:, 1], color='cyan', linestyle='-', label='GT Rotation Angle')
        plt.plot(range(len(test_dataloader)), results[:, 2], color='blue', linestyle='-', label='Difference')
        plt.title(f'Test for {weight_name}')
        plt.xlabel("Datapoint")
        plt.ylabel("Degree")
        plt.legend()
        
        if show is True:
            plt.show()
        else:
            file_name = FIGURES_PATH + f'{weight_name.split(".pth")[0]}_test_with_{TEST_DATASET_NAME}_{TEST_DATASET_TYPE}'
            plt.savefig(file_name + '.png')
            plt.close()

            np.savetxt(file_name + '.csv', results, delimiter=',')

if __name__ == '__main__':

    model = FiveResNet18MLP5_7()

    # Preprocess for images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if torch.cuda.is_available():
        
        test_dataset = SPOTDataLoader(
            root_dir = TEST_PATH,
            goal_folder = GOAL_PATH,
            labels_file = LABEL_PATH,
            transform = data_transforms
        )
        DEVICE = 'cuda'
        print('Cuda')

    else:
        test_dataset = SPOTDataLoader(
            root_dir = TEST_PATH,
            goal_folder = GOAL_PATH,
            labels_file = LABEL_PATH,
            transform = data_transforms
        )
        DEVICE = 'cpu'
        print('CPU')

    weight_name = WEIGHT_NAME
    test_model(test_dataset, model, weight_name, device=DEVICE, draw=True, show=False)