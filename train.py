import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from groundingdino.util.inference import load_model
from DataLoader import SPOT_SingleStep_DataLoader
from DinoMLP import DINOCrossAttentionMLP
from plot import plot_graph
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_least_used_gpu():
    # If you have multiple GPUs and want to pick one with max free memory
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append(free)
    best_gpu = max(range(torch.cuda.device_count()), key=lambda i: gpu_free_memory[i])
    return best_gpu


#############################################
# Main training function
#############################################
def train():
    # If you have an existing checkpoint, set CONTINUE > 0
    CONTINUE = 0

    # dataset info
    DATASET_NAMES = ['map01_01', 'map01E_01']
    WEIGHT_FOLDER_NAME = 'lr1e-3'
    MODEL_NAME = 'DinoMLP'

    SCRIPT_PATH = os.getcwd() 
    DATASET_PATHS = []
    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join('/data/lee04484/SPOT_Real_World_Dataset/', dataset_name)
        if not os.path.exists(dataset_path):
            print(f'Dataset {dataset_name} does not exist!')
            return
        DATASET_PATHS.append(dataset_path)

    WEIGHT_PATH = os.path.join(SCRIPT_PATH, f'weights/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}')
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(WEIGHT_PATH)

    FIGURE_PATH = os.path.join(SCRIPT_PATH, f'Results/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}')
    if not os.path.exists(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if torch.cuda.is_available():
        best_gpu = get_least_used_gpu()
        DEVICE = f'cuda:{best_gpu}'
        print(f'Using GPU: {DEVICE}')
    else:
        DEVICE = 'cpu'
        print('Using CPU!')

    # Hyperparameters
    loss_fn = torch.nn.MSELoss()
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    WEIGHT_SAVING_STEP = 10
    LOSS_SCALE = 1e3
    TOLERANCE = 1e-2

    hyper_params_path = os.path.join(WEIGHT_PATH, 'hyper_params')
    hyper_params = {'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE, 'LOSS_SCALE': LOSS_SCALE, 'TOLERANCE': TOLERANCE}
    np.savez(hyper_params_path, **hyper_params)
    print(f'BATCH_SIZE={BATCH_SIZE}, LR={LEARNING_RATE}, LOSS_SCALE={LOSS_SCALE}, TOLERANCE={TOLERANCE}')

    # -------------------------------------------------------------
    # Load base_model 
    # -------------------------------------------------------------
    config_file = "GroundingDINO_SwinT_OGC.py"
    weight_file = "weights/groundingdino_swint_ogc.pth"

    # === Model, Optimizer, and Scheduler ===
    model = DINOCrossAttentionMLP(
        config_file=config_file,
        weight_file=weight_file,
        device=DEVICE ,
        num_cameras=5,
        embed_dim=256,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    epoch = CONTINUE + 1
    training_loss = 1e6
    training_total_loss = 0
    training_losses = []
    tracking_losses_path = os.path.join(FIGURE_PATH, 'training_losses.npy')
    accuracies = []
    accuracies_path = os.path.join(FIGURE_PATH, 'accuracies.npy')

    if CONTINUE > 1:
        last_weight_path = os.path.join(WEIGHT_PATH, f'epoch_{CONTINUE}.pth')
        model.load_state_dict(torch.load(last_weight_path, map_location=DEVICE))
        print('Loaded checkpoint!')

        # If you previously saved training_losses & accuracies
        training_losses = list(np.load(tracking_losses_path))[:CONTINUE]
        accuracies = list(np.load(accuracies_path))[:CONTINUE]

    
    train_dataset = SPOT_SingleStep_DataLoader(
        dataset_dirs=DATASET_PATHS,
        transform=data_transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    while training_loss > (TOLERANCE ** 2) * LOSS_SCALE:
        model.train()
        running_loss = 0.0

        for current_images, goal_images, labels in train_dataloader:
            current_images = current_images.to(DEVICE)
            goal_images = goal_images.to(DEVICE)
            labels = labels.to(DEVICE)
            text_prompts = ["green chair." for _ in range(current_images.size(0))]

            for i, prompt in enumerate(text_prompts):
                if not isinstance(prompt, str):
                    print(f"Element {i} is not a string: {prompt} (type: {type(prompt)})")

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            non_string = [(i, type(prompt)) for i, prompt in enumerate(text_prompts) if not isinstance(prompt, str)]
            print(non_string)


            output = model(current_images, goal_images, text_prompts)
            loss = loss_fn(output, labels.float()) * LOSS_SCALE

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            running_loss += loss.item()
            torch.cuda.empty_cache()

        
        training_loss = running_loss / len(train_dataloader)

        # Moving average
        training_total_loss += training_loss * 5
        training_average_loss = training_total_loss / (len(training_losses) + 5)
        training_total_loss = training_average_loss * (len(training_losses) + 1)

        training_losses.append([training_loss, training_average_loss])

        print(f"[Epoch {epoch}] Loss: {training_loss:.6f}, Avg Loss: {training_average_loss:.6f}", end='; ')
        np.save(tracking_losses_path, training_losses)

        scheduler.step(training_loss)

        # Save model weights every 10 epochs
        if epoch % WEIGHT_SAVING_STEP == 0:
            save_path = os.path.join(WEIGHT_PATH, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved Weights -> {save_path}", end='; ')

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            num_correct, num_total = 0, 0
            for current_images, goal_images, labels in train_dataloader:
                current_images = current_images.to(DEVICE)
                goal_images = goal_images.to(DEVICE)
                text_prompts = ["green chair." for _ in range(current_images.size(0))]
                labels = labels.to(DEVICE)

                preds = model(current_images, goal_images, text_prompts)

                for i in range(preds.size(0)):
                    diff = float(torch.sum(torch.abs(preds[i] - labels[i])))
                    num_total += 1
                    if diff < TOLERANCE:
                        num_correct += 1

            train_accuracy = (num_correct / num_total) * 100
            accuracies.append(train_accuracy)
            print(f"Train Accuracy : {train_accuracy:.2f}%")
            np.save(accuracies_path, accuracies)

        epoch += 1

    print("Finished Training!")
    epoch -= 1

    last_path = os.path.join(WEIGHT_PATH, f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), last_path)
    print(f"Saved Final Weights -> {last_path}")
    plot_graph(training_losses, accuracies, weight_save_step=WEIGHT_SAVING_STEP, figure_path=FIGURE_PATH, end_plot=epoch)

if __name__ == "__main__":
    train()
