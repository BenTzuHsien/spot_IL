import torch, os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from SPOT_SingleStep_DataLoader import SPOT_SingleStep_DataLoader
from Resnet18MLP5 import SharedResNet18MLP5
from plot_graph import plot_graph

CONTINUE = 0   # Start fresh at 0

# Setup Destination
MODEL_NAME = 'ResNet18MLP5'
DATASET_NAMES = ['map01_01', 'map01E1_01', 'map01E2_01', 'map01E3_01']
DATASET_DIR = '/data/lee04484/SPOT_Real_World_Dataset/'

# Hyper Parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# Training Parameters
WEIGHT_SAVING_STEP = 20
LOSS_SCALE = 1e3

# Validation Parameter
TOLERANCE = 1e-2

def get_least_used_gpu():
    
    # Get available memory for each GPU
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append(free)

    least_used_gpu = max(range(torch.cuda.device_count()), key=lambda i: gpu_free_memory[i])
    return least_used_gpu

# Preprocess for images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':

    # Setup Dataset Path
    DATASET_PATHS = []
    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join(DATASET_DIR, f'{dataset_name}')
        if not os.path.exists(dataset_path):
            print(f'Dataset {dataset_name} does not exist !')
            exit()
        DATASET_PATHS.append(dataset_path)

    # Setup Weight & Result Saving Path
    SCRIPT_PATH = os.path.dirname(__file__)
    WEIGHT_FOLDER_NAME = f'lr{LEARNING_RATE:.0e}'
    WEIGHT_PATH = os.path.join(SCRIPT_PATH, f'weights/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}/')
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(WEIGHT_PATH)
    FIGURE_PATH = os.path.join(SCRIPT_PATH, f'Results/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}/')
    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    if torch.cuda.is_available():
        least_used_gpu = get_least_used_gpu()
        DEVICE = f'cuda:{least_used_gpu}'
        print(f'cuda:{least_used_gpu}')

    else:
        DEVICE = 'cpu'
        print('CPU')

    # Saving Hyper Param
    hyper_params_path = os.path.join(WEIGHT_PATH, 'hyper_params')
    hyper_params = {'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE, 'LOSS_SCALE': LOSS_SCALE, 'TOLERANCE': TOLERANCE}
    print(f'BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, LOSS_SCALE: {LOSS_SCALE}, TOLERANCE: {TOLERANCE}')
    np.savez(hyper_params_path, **hyper_params)

    # Setup Model
    model = SharedResNet18MLP5().to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tracking Parameters
    training_total_loss = 0
    training_losses = []   #[training_loss training_average_loss]
    tracking_losses_path = os.path.join(FIGURE_PATH, 'training_losses.npy')
    accuracies = []
    accuracies_path = os.path.join(FIGURE_PATH, 'accuracies.npy')

    if CONTINUE > 1:
        lastest_weight_path = os.path.join(WEIGHT_PATH, 'epoch_' + str(CONTINUE) + '.pth')
        model.load_state_dict(torch.load(lastest_weight_path))
        print('Weight Loaded!')
        training_losses = list(np.load(tracking_losses_path))[:CONTINUE]
        tracking_losses_path = os.path.join(FIGURE_PATH, 'new_training_losses.npy')
        accuracies = list(np.load(accuracies_path))[:CONTINUE]
        accuracies_path = os.path.join(FIGURE_PATH, 'new_accuracies.npy')
        print('Parameter Loaded!')

    train_dataset = SPOT_SingleStep_DataLoader(
            dataset_dirs = DATASET_PATHS,
            transform = data_transforms
        )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # Train Model
    for epoch in range(CONTINUE, 1500):
        
        model.train()
        running_loss = 0.0
        
        for current_images, goal_image, labels in train_dataloader:

            current_images = current_images.to(DEVICE)
            goal_image = goal_image.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(current_images, goal_image)
            loss = loss_fn(output, labels) * LOSS_SCALE

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            torch.cuda.empty_cache()

        training_loss = running_loss / len(train_dataloader)

        # Moving Average
        training_total_loss += training_loss * 5
        training_average_loss = training_total_loss / (len(training_losses) + 5)
        training_total_loss = training_average_loss * (len(training_losses) + 1)

        # Save training loss
        training_losses.append([training_loss, training_average_loss])
        print(f'Epoch {epoch + 1}, Loss: {training_losses[epoch][0]:.6f}, Average Loss: {training_losses[epoch][1]:.6f}', end='; ')
        np.save(tracking_losses_path, training_losses)

        if ((epoch + 1) % WEIGHT_SAVING_STEP) == 0:
            weight_save_path = os.path.join(WEIGHT_PATH, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), weight_save_path)
            print('Save Weights', end='; ')

        # Valid Model
        model.eval()
        with torch.no_grad():

            num_correct, num_total = 0, 0
            for current_images, goal_image, labels in train_dataloader:

                current_images = current_images.to(DEVICE)
                goal_image = goal_image.to(DEVICE)
                labels = labels.to(DEVICE)

                output = model(current_images, goal_image)
                for i in range(output.shape[0]):
                    loss = 0
                    for j in range(output.shape[1]):
                        loss += abs(output[i][j] - labels[i][j]).item()
                    num_total += 1
                    if loss < TOLERANCE:
                        num_correct += 1
            train_accuracy = (num_correct / num_total) * 100

            accuracies.append(train_accuracy)
            print(f'Train Accuracy {accuracies[epoch]:.2f}%')
            np.save(accuracies_path, accuracies)

    print('Finished Training !')

    # Save last weight
    weight_save_path = os.path.join(WEIGHT_PATH, 'epoch_' + str(epoch + 1) + '.pth')
    torch.save(model.state_dict(), weight_save_path)
    print('Save Last Weights')

    # Plot Training Loss and Accuracies graphs
    plot_graph(training_losses, accuracies, weight_save_step=WEIGHT_SAVING_STEP, figure_path=FIGURE_PATH, end_plot=(epoch + 1))
