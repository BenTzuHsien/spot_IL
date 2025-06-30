import os, glob, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from GsamMlp5_discretized_fast import GsamMlp5
import numpy as np
from plot_graph import plot_graph  


CONTINUE = 0   # Start fresh at 0

# Setup Destination
MODEL_NAME = 'GSAMMLP5'
EMBED_DIR  = os.path.join(os.getcwd(), 'gsam_features')  

# Hyper Parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Training Parameters
WEIGHT_SAVING_STEP = 20

# Validation Parameter
TOLERANCE = 1e-2

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpotEmbedsDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, '*.pt')))
        if not self.files:
            raise RuntimeError(f'No .pt files found in {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(
            self.files[idx],
            map_location='cpu',     
            weights_only=True      
        )
        return (
            d['current_embeddings'].float(),
            d['goal_embeddings'].float(),
            d['labels']
        )

    

if __name__ == '__main__':

    # Setup Weight & Result Saving Path
    SCRIPT_PATH = os.path.dirname(__file__)
    WEIGHT_FOLDER_NAME = 'cleanup'
    WEIGHT_PATH = os.path.join(SCRIPT_PATH, f'weights/{MODEL_NAME}')
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(WEIGHT_PATH)
    FIGURE_PATH = os.path.join(SCRIPT_PATH, f'Results/{MODEL_NAME}')
    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    model = GsamMlp5().to(DEVICE)
    
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


    train_dataset = SpotEmbedsDataset(EMBED_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    for epoch in range(CONTINUE, 1000):

        model.train()
        running_loss = 0.0
    
        for current_feat, goal_feat, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{1000}"):
            current_feat = current_feat.to(DEVICE)
            goal_feat = goal_feat.to(DEVICE)
            labels = labels.to(DEVICE)
        
            optimizer.zero_grad()
            outputs = model(current_feat, goal_feat)
            outputs = outputs.permute(0, 2, 1)   # To accomadate how CrossEnropyLoss function accept as input (Batch_size, Num_classes, ...)
            loss = loss_fn(outputs, labels)

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


            

