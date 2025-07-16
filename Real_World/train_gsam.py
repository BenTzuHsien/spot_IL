import os, glob, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from plot_graph import plot_graph  
import math   
from gsam_mlp5_bi2 import GsamMlp5Bi2
from torch.cuda.amp import GradScaler 


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


CONTINUE = 0   # Start fresh at 0

# Setup Destination
MODEL_NAME = 'bi2_fully_masked_CONV'
EMBED_DIR  = os.path.join(os.getcwd(), 'GSAM_feats_fully_masked')  

# Hyper Parameters
BATCH_SIZE =  32
LEARNING_RATE = 1e-4
PROMPT_TXT = "green chair."
import torch
scaler = torch.cuda.amp.grad_scaler.GradScaler() 


# Training Parameters
WEIGHT_SAVING_STEP = 50

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
    
    if torch.cuda.is_available():
        DEVICE = "cuda:0"   
    else:
        DEVICE = 'cpu'
        print('CPU')
    
    # WE USE SAVED EMBEDDINGS !
    model = GsamMlp5Bi2(use_gsam=False).to(device=DEVICE, dtype=torch.float)
    
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

    for epoch in range(CONTINUE, 500):

        model.train()
        running_loss = 0.0
    
        for current_feat, goal_feat, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{500}"):
            current_feat = current_feat.to(DEVICE, non_blocking = True)
            goal_feat = goal_feat.to(DEVICE, non_blocking = True)
            labels = labels.to(DEVICE, non_blocking = True)
        
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                model.set_goal(goal_feat, PROMPT_TXT)  
                outputs, _ = model(current_feat)
                outputs    = outputs.permute(0, 2, 1)
                loss       = loss_fn(outputs, labels)


            scaler.scale(loss).backward()
            scaler.step(optimizer)  
            scaler.update()         
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
            for current_feat, goal_feat, labels in train_dataloader:

                current_feat = current_feat.to(DEVICE, non_blocking = True)
                goal_feat = goal_feat.to(DEVICE, non_blocking = True)
                labels = labels.to(DEVICE, non_blocking = True)

                model.set_goal(goal_feat, "green chair.")
                outputs, attention = model(current_feat)
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


            

