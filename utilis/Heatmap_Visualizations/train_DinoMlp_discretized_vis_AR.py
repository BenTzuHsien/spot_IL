import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from SPOT_SingleStep_Discredtized_DataLoader import SPOT_SingleStep_Discretized_DataLoader
from DinoMLP5_discretized import DinoMLP5_discretized
from plot_graph import plot_graph



# --------------- Attention Rollout Implementation sections --------------------
def get_all_selfattentions(vit, x):
    # Empty list to accumulate attention matrices for each Transformer Block
    attn_tensors = []
    
    def _hook(module, inp, _out):
        # N : number of tokens 
        # N = 261 = 256 + 4 (register tokens) + 1 [CLS]
        # C : embedding dimension (384)
        B, N, C = inp[0].shape
       
        # ---- Recomputing QKV projections ---
        # Linear projection layer calculates Q, K, V for us

        # shape is : [B, N , 3*C] concatenated Q, K, V
        # reshape result : [B, N, 3, num_heads, head_dim]
        # permute : [3, B, num_heads, N, head_dim]

        # number of attention heads : 6
        # per-head embedding dimension : 64
        qkv = module.qkv(inp[0]) \
                .reshape(B, N, 3, module.num_heads, C // module.num_heads) \
                .permute(2, 0, 3, 1, 4)
        

        # Q : [B, num_heads, N, head_dim]
        # K : [B, num_heads, N, head_dim]
        q, k = qkv[0], qkv[1]
        
        # attention calculation step
        # attention score between every pair of tokens
        # attn shape : [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        if getattr(module, "attn_bias", None) is not None:
            attn = attn + module.attn_bias

        # softmax step 
        attn_tensors.append(attn.softmax(-1).detach())

    handles = [blk.attn.register_forward_hook(_hook) for blk in vit.blocks]
    vit.eval()

    # Running a single forward pass on input  (triggers the hooks and fills attn_tensors)
    vit(x)  

    for h in handles:
        h.remove()

    # Return list of attention tensors for Attention Rollout
    # num_layers = 12
    # has 12 layers each w/ shape : [B, num_heads, N, N]
    return attn_tensors 


def compute_rollout(attn_list, head_fusion="max", discard_ratio=0.9,
                    self_weight=0.2, num_reg=4):
    
    # stacking attention matrices
    # attn: [L=12, B, head_dim, N, N]
    attn = torch.stack(attn_list)  

    # Head fusion
    # removing head_dim
    # attn: [L=12, B, N, N]
    if head_fusion == "mean":
        attn = attn.mean(dim=2)
    elif head_fusion == "max":
        attn = attn.max(dim=2).values
    elif head_fusion == "min":
        attn = attn.min(dim=2).values

    L, B, N, _ = attn.shape
    # Discard lowest attentions
    if discard_ratio > 0:
        flat = attn.flatten(2)
        thr = torch.quantile(flat, discard_ratio, dim=2).view(L, B, 1, 1)
        attn = torch.where(attn < thr, 0.0, attn)

    # Add scaled identity & normalize rows
    eye = torch.eye(N, device=attn.device)
    attn = attn + self_weight * eye.view(1, 1, N, N)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Attention Rollout
    rollout = eye.unsqueeze(0).repeat(B, 1, 1)
    for A in attn.flip(0): # from last layer to first
        # rollout shape : [B, N, N]
        rollout = A @ rollout

    # 0: CLS token index
    # 1 + num_reg = 5: start of patch tokens
    cls2patch = rollout[:, 0, 1 + num_reg:]
    # cls2patch shape : [B, 256]
    # 1 attention score per patch 
    return cls2patch


def visualize_and_save(frames, cls2patch, epoch, save_dir, grid_side, mode):
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    imgs = (frames * std + mean).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    os.makedirs(save_dir, exist_ok=True)

    n_vis = imgs.shape[0]
    fig, axes = plt.subplots(2, n_vis, figsize=(3 * n_vis, 6))

    for i in range(n_vis):
        # Normalize heatmap to [0,1]
        heat = cls2patch[i].reshape(grid_side, grid_side).cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat = cv2.resize(heat, (imgs.shape[2], imgs.shape[1]), cv2.INTER_CUBIC)
        # Overlay heatmap
        axes[0, i].imshow(imgs[i], alpha=0.6)
        axes[0, i].imshow(heat, cmap='hot', vmin=0, vmax=1, alpha=0.4)
        axes[0, i].axis('off')
        # Show original image below
        axes[1, i].imshow(imgs[i])
        axes[1, i].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"AR_epoch_{epoch+1}_{mode}.png"), dpi=150)
    plt.close(fig)

# -------------------------------------------------------------------------
CONTINUE = 0   # Start fresh at 0

# Setup Destination
MODEL_NAME = 'DinoMLP_discretized'
DATASET_NAMES = ['map01_01a', 'map01_01b', 'map01_02a', 'map01_02b', 'map01_03a', 'map01_03b']
DATASET_DIR = '/data/shared_data/SPOT_Real_World_Dataset/cleanup_dataset/'

# Hyper Parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# Training Parameters
WEIGHT_SAVING_STEP = 20
SAVE_VIS = 1

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

if __name__ == "__main__":

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
        model = DinoMLP5_discretized().to(primary_device)
        model = torch.nn.DataParallel(model, device_ids=top_gpus)
        DEVICE = primary_device  # For consistency in moving tensors to device
    else:
        DEVICE = 'cpu'
        print('Using CPU')
        model = DinoMLP5_discretized().to(DEVICE)

    vit = (model.module if hasattr(model, 'module') else model).shared_trunk
    n_reg = getattr(vit, "num_register_tokens", 4)

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
    head_modes = ["mean", "max", "min"]
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

        if ((epoch + 1) % SAVE_VIS) == 0:
            model.eval()
            with torch.no_grad():
                batch_imgs, _, _ = next(iter(train_dataloader))
                sample0 = batch_imgs[0].to(DEVICE) 
                attn_list = get_all_selfattentions(vit, sample0)
                grid_side = int(math.sqrt(attn_list[0].shape[-1] - 1 - n_reg))
                for mode in head_modes:
                    cls2patch = compute_rollout(
                        attn_list,
                        head_fusion=mode,
                        discard_ratio=0.9,
                        self_weight=0.2,
                        num_reg=n_reg
                    )
                    save_subdir = os.path.join(FIGURE_PATH, f"sample0_{mode}")
                    visualize_and_save(
                        sample0, cls2patch, epoch,
                        save_subdir, grid_side, mode
                    )
        print()

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

