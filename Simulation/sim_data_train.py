from torch.utils.data import DataLoader
import torch, os
from torchvision import transforms
from spotdatasetloader import SPOTDataLoader
from FiveResNet18MLP5_7 import FiveResNet18MLP5_7
from Simulation.plot_graph import plot_graph
import numpy as np
from sklearn.model_selection import KFold

# Setup Destination
DATASET_NAME = 'mixed'
WEIGHT_FOLDER_NAME = 'lr1e-6_full_output'

DATASET_INIRIAL_PATH = os.getcwd() + f'/dataset_{DATASET_NAME}/'
TRAIN_PATH = DATASET_INIRIAL_PATH + 'train/'
GOAL_PATH = DATASET_INIRIAL_PATH + 'goal/'
LABEL_PATH = TRAIN_PATH + 'labels.npy'
WEIGHT_PATH = os.getcwd() + f'/weights/FiveResNet18MLP5_{DATASET_NAME}/{WEIGHT_FOLDER_NAME}/'
if not os.path.exists(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)
FIGURE_PATH = os.getcwd() + f'/Results/FiveResNet18MLP5_{DATASET_NAME}/{WEIGHT_FOLDER_NAME}/'
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# KFold Parameters
NUM_FOLD = 3
k_fold = KFold(NUM_FOLD, shuffle=True)
CONTINUE = [0] * NUM_FOLD   # Start from beginning, use 0

# Preprocess for images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if torch.cuda.is_available():
    
    train_dataset = SPOTDataLoader(
        root_dir = TRAIN_PATH,
        goal_folder = GOAL_PATH,
        labels_file = LABEL_PATH,
        transform = data_transforms
    )
    DEVICE = 'cuda'
    print('Cuda')

else:
    train_dataset = SPOTDataLoader(
        root_dir = TRAIN_PATH,
        goal_folder = GOAL_PATH,
        labels_file = LABEL_PATH,
        transform = data_transforms
    )
    DEVICE = 'cpu'
    print('CPU')

# Hyper Parameters
loss_fn = torch.nn.MSELoss()
BATCH_SIZE = 16
LEARNING_RATE = 1e-6

# Training Parameters
WEIGHT_SAVING_STEP = 50
LOSS_SCALE = 1e3

# Validation Parameter
TOLERANCE = 1e-4

# Saving Hyper Param
hyper_params_path = WEIGHT_PATH + 'hyper_params'
hyper_params = {'NUM_FOLD': NUM_FOLD, 'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE, 'LOSS_SCALE': LOSS_SCALE, 'TOLERANCE': TOLERANCE}
np.savez(hyper_params_path, **hyper_params)

for fold, (train_ids, valid_ids) in enumerate(k_fold.split(train_dataset)):
    print(f'FOLD {fold}')
    fold_path = WEIGHT_PATH + 'fold_' + str(fold) + '/'
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)

    # Setup Model
    model = FiveResNet18MLP5_7().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tracking Parameters
    epoch = CONTINUE[fold] + 1
    training_loss = 1e6
    training_total_loss = 0
    training_losses = []   #[training_loss training_average_loss]
    tracking_losses_path = fold_path + 'training_losses.npy'
    accuracies = []   #[train_accuracy valid_accuracy]
    accuracies_path = fold_path + 'accuracies.npy'

    if CONTINUE[fold] > 1:
        model.load_state_dict(torch.load(fold_path + 'epoch_' + str(CONTINUE[fold]) + '.pth'))
        print('Weight Loaded!')
        training_losses = list(np.load(tracking_losses_path))[:CONTINUE[fold]]
        accuracies = list(np.load(accuracies_path))[:CONTINUE[fold]]
        print(f'Fold {fold} Parameter Loaded!')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=valid_subsampler)

    # Train Model
    model.train()
    while training_loss > ((TOLERANCE ** 2) * LOSS_SCALE):
        
        running_loss = 0.0
        
        for current_images, goal_images, labels in train_dataloader:
            
            optimizer.zero_grad()
            output = model(current_images, goal_images)
            loss = loss_fn(output, labels) * LOSS_SCALE

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        training_loss = running_loss / len(train_dataloader)

        # Moving Average
        training_total_loss += training_loss * 5
        training_average_loss = training_total_loss / (len(training_losses) + 5)
        training_total_loss = training_average_loss * (len(training_losses) + 1)

        # Save training loss
        training_losses.append([training_loss, training_average_loss])
        print(f'Epoch {epoch}, Loss: {training_losses[epoch - 1][0]:.6f}, Average Loss: {training_losses[epoch - 1][1]:.6f}', end='; ')
        np.save(tracking_losses_path, training_losses)

        if (epoch % WEIGHT_SAVING_STEP) == 0:
            torch.save(model.state_dict(), (fold_path + 'epoch_' + str(epoch) + '.pth'))
            print('Save Weights', end='; ')

        # Valid Model
        model.eval()
        with torch.no_grad():

            num_correct, num_total = 0, 0
            for current_images, goal_images, labels in train_dataloader:
                output = model(current_images, goal_images)
                for i in range(len(output)):
                    loss = 0
                    for j in range(7):
                        loss += abs(output[i][j] - labels[i][j]).item()
                    num_total += 1
                    if loss < TOLERANCE:
                        num_correct += 1
            train_accuracy = (num_correct / num_total) * 100

            num_correct, num_total = 0, 0
            for current_images, goal_images, labels in valid_dataloader:
                output = model(current_images, goal_images)
                for i in range(len(output)):
                    loss = 0
                    for j in range(7):
                        loss += abs(output[i][j] - labels[i][j]).item()
                    num_total += 1
                    if loss < TOLERANCE:
                        num_correct += 1
            valid_accuracy = (num_correct / num_total) * 100

            accuracies.append([train_accuracy, valid_accuracy])
            print(f'Train Accuracy {accuracies[epoch - 1][0]:.2f}%, Valid Accuracy: {accuracies[epoch - 1][1]:.2f}%')
            np.save(accuracies_path, accuracies)

            epoch += 1

    print(f'Finished Training fold {fold}')
    epoch -= 1

    # Save last weight
    torch.save(model.state_dict(), (WEIGHT_PATH + 'fold_' + str(fold) + '/epoch_' + str(epoch) + '.pth'))
    print('Save Last Weights')

    # Plot Training Loss and Accuracies graphs
    plot_graph(training_losses, accuracies, FIGURE_PATH, fold, end_plot=epoch)
