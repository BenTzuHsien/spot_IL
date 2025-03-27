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

def get_top_available_gpus(n=3):
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append((free, i))
    top_gpus = sorted(gpu_free_memory, reverse=True)[:n]
    return [gpu_idx for free, gpu_idx in top_gpus]

def get_least_used_gpu():
    gpu_free_memory = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        gpu_free_memory.append(free)
    best_gpu = max(range(torch.cuda.device_count()), key=lambda i: gpu_free_memory[i])
    return best_gpu

def train():
    CONTINUE = 0
    DATASET_NAMES = ['map01_01a', 'map01_01b', 'map01_02a', 'map01_02b', 'map01_03a', 'map01_03b']
    WEIGHT_FOLDER_NAME = 'lr1e-3'
    MODEL_NAME = 'DinoMLP'
    SCRIPT_PATH = os.getcwd() 
    DATASET_PATHS = []
    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join('/data/shared_data/SPOT_Real_World_Dataset/', dataset_name)
        if not os.path.exists(dataset_path):
            print(f'Dataset {dataset_name} does not exist!')
            return
        DATASET_PATHS.append(dataset_path)
    WEIGHT_PATH = os.path.join(SCRIPT_PATH, f'weights/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}')
    os.makedirs(WEIGHT_PATH, exist_ok=True)
    FIGURE_PATH = os.path.join(SCRIPT_PATH, f'Results/{MODEL_NAME}_{"_".join(DATASET_NAMES)}/{WEIGHT_FOLDER_NAME}')
    os.makedirs(FIGURE_PATH, exist_ok=True)

    # Change image resize to 800x800 (GroundingDINO is typically configured for high-res images)
    data_transforms = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    config_file = "/home/mahmu059/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weight_file = "weights/groundingdino_swint_ogc.pth"

    if torch.cuda.is_available():
        least_used_gpu = get_least_used_gpu()
        primary_device = f'cuda:{least_used_gpu}'
        print(primary_device)
        model = DINOCrossAttentionMLP(
            config_file=config_file,
            weight_file=weight_file,
            device=primary_device,
            num_cameras=5,
            embed_dim=256,
        )
        model.to(primary_device)
        DEVICE = primary_device
    else:
        DEVICE = 'cpu'
        print('Using CPU!')
        model = DINOCrossAttentionMLP(
            config_file=config_file,
            weight_file=weight_file,
            device=DEVICE,
            num_cameras=5,
            embed_dim=256,
        )

    loss_fn = torch.nn.MSELoss()
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_SAVING_STEP = 10
    LOSS_SCALE = 1e3
    TOLERANCE = 1e-2

    hyper_params = {'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE,
                    'LOSS_SCALE': LOSS_SCALE, 'TOLERANCE': TOLERANCE}
    np.savez(os.path.join(WEIGHT_PATH, 'hyper_params'), **hyper_params)
    print(f'BATCH_SIZE={BATCH_SIZE}, LR={LEARNING_RATE}, LOSS_SCALE={LOSS_SCALE}, TOLERANCE={TOLERANCE}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    epoch = CONTINUE + 1
    training_losses = []
    tracking_losses_path = os.path.join(FIGURE_PATH, 'training_losses.npy')
    
    train_dataset = SPOT_SingleStep_DataLoader(
        dataset_dirs=DATASET_PATHS,
        transform=data_transforms
    )
    # If issues persist, try setting num_workers=0 for debugging.
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    while True:
        model.train()
        running_loss = 0.0
        for current_images, goal_images, labels in train_dataloader:
            current_images = current_images.to(DEVICE).contiguous()

            if goal_images.dim() == 4:  # shape [B, C, H, W]
                goal_images = goal_images.unsqueeze(1).expand(-1, 5, -1, -1, -1).contiguous()

            goal_images = goal_images.to(DEVICE).contiguous()
            labels = labels.to(DEVICE)
            text_prompts = ["green chair." for _ in range(current_images.size(0))]

            optimizer.zero_grad()
            try:
                output = model(current_images, goal_images, text_prompts)
                torch.cuda.synchronize()
            except Exception as e:
                print("Error during forward pass:")
                print(e)
                print("Current images shape:", current_images.shape)
                print("Goal images shape:", goal_images.shape)
                raise e

            loss = loss_fn(output, labels.float()) * LOSS_SCALE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            running_loss += loss.item()
            torch.cuda.empty_cache()
        
        training_loss = running_loss / len(train_dataloader)
        training_losses.append(training_loss)
        np.save(tracking_losses_path, training_losses)
        print(f"[Epoch {epoch}] Loss: {training_loss:.6f}")
        scheduler.step(training_loss)
        if epoch % WEIGHT_SAVING_STEP == 0:
            save_path = os.path.join(WEIGHT_PATH, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved Weights -> {save_path}")
        if training_loss < (TOLERANCE ** 2) * LOSS_SCALE:
            break
        epoch += 1

    print("Finished Training!")
    final_save_path = os.path.join(WEIGHT_PATH, f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved Final Weights -> {final_save_path}")
    plot_graph(training_losses, [], weight_save_step=WEIGHT_SAVING_STEP, figure_path=FIGURE_PATH, end_plot=epoch)

if __name__ == "__main__":
    train()
