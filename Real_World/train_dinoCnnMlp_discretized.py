import torch, os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from SPOT_SingleStep_Discredtized_DataLoader import SPOT_SingleStep_Discretized_DataLoader
from models.DinoCnn2MLP3_discretized import DinoCnn2MLP3_discretized
from plot_graph import plot_graph

CONTINUE = 0   # Start fresh at 0

# Setup Destination
MODEL_NAME = 'DinoCnnMLP_discretized'
DATASET_NAMES = ['map01_01a', 'map01_01b', 'map01_02a', 'map01_02b', 'map01_03a', 'map01_03b']
DATASET_DIR = '/data/lee04484/SPOT_Real_World_Dataset/cleanup_dataset/'

# Hyper Parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# Training Parameters
WEIGHT_SAVING_STEP = 20

# Validation Parameter
TOLERANCE = 1e-2

def get_top_available_gpus(n=3):
    # Get available memory for each GPU and return the indices of the top n GPUs
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append((free, i))
    top_gpus = sorted(gpu_free_memory, reverse=True)[:n]
    return [gpu_idx for free, gpu_idx in top_gpus]

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
            print(f'Dataset {dataset_name} does not exist!')
            exit()
        DATASET_PATHS.append(dataset_path)

    # Setup Weight & Result Saving Path
    SCRIPT_PATH = os.path.dirname(__file__)
    WEIGHT_FOLDER_NAME = 'cleanup'
    WEIGHT_PATH = os.path.join(SCRIPT_PATH, f'weights/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}/')
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(WEIGHT_PATH)
    FIGURE_PATH = os.path.join(SCRIPT_PATH, f'Results/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}/')
    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    # Multi-GPU Setup
    if torch.cuda.is_available():
        top_gpus = get_top_available_gpus(2)
        primary_device = f'cuda:{top_gpus[0]}'
        print(f'Using GPUs: {top_gpus}')
        model = DinoCnn2MLP3_discretized().to(primary_device)
        model = torch.nn.DataParallel(model, device_ids=top_gpus)
        DEVICE = primary_device  # For consistency in moving tensors to device
    else:
        DEVICE = 'cpu'
        print('Using CPU')
        model = DinoCnn2MLP3_discretized().to(DEVICE)

    # Saving Hyper Param
    hyper_params_path = os.path.join(WEIGHT_PATH, 'hyper_params')
    hyper_params = {'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE, 'TOLERANCE': TOLERANCE}
    print(f'BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, TOLERANCE: {TOLERANCE}')
    np.savez(hyper_params_path, **hyper_params)

    # Setup loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tracking Parameters
    training_total_loss = 0
    training_losses = []   # [training_loss, training_average_loss]
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

    train_dataset = SPOT_SingleStep_Discretized_DataLoader(
            dataset_dirs=DATASET_PATHS,
            transform=data_transforms
        )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # Train Model
    for epoch in range(CONTINUE, 1000):

        model.train()
        running_loss = 0.0

        for current_images, goal_image, labels in train_dataloader:

            current_images = current_images.to(DEVICE)
            goal_image = goal_image.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(current_images, goal_image)
            
            outputs = outputs.permute(0, 2, 1)   # To accomadate how CrossEnropyLoss function accept as input (Batch_size, Num_classes, ...)
            loss = loss_fn(outputs, labels)

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
            # When using DataParallel, save the model.module's state dict
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

                outputs = model(current_images, goal_image)
                prediction = torch.argmax(outputs, dim = 2)

                prediction_mask = torch.all(prediction == labels, dim=1)
                num_total += prediction_mask.shape[0]
                num_correct += prediction_mask.sum().item()

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
