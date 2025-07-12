import os, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


from SPOT_SingleStep_Discredtized_DataLoader import SPOT_SingleStep_Discretized_DataLoader
from backbones.grounded_sam2 import GroundedSAM2

# Paths
DATASET_NAMES = ['map1']
DATASET_DIR = '/data/shared_data/SPOT_Real_World_Dataset/'

FEATURES_DIR = os.path.join(os.getcwd(), 'GSAM_feats_fully_masked') 
os.makedirs(FEATURES_DIR, exist_ok=True)

# Hyperameters
BATCH_SIZE   = 1
PROMPT_TXT = "green chair."


# Device setup 

# change device here 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def get_least_used_gpu(devices=range(torch.cuda.device_count())):
    # Get available memory for the specified GPUs
    gpu_free_memory = {i: torch.cuda.mem_get_info(i)[0] for i in devices}
    # Pick the GPU with the most free memory
    return max(gpu_free_memory, key=gpu_free_memory.get)

DEVICE = "cuda:0"   


# Dataloading
DATASET_PATHS = []
for dataset_name in DATASET_NAMES:
    dataset_path = os.path.join(DATASET_DIR, f'{dataset_name}')
    if not os.path.exists(dataset_path):
        print(f'Dataset {dataset_name} does not exist!')
        exit()
    DATASET_PATHS.append(dataset_path)


data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
           #  transforms.ToTensor(),         
])

dataset = SPOT_SingleStep_Discretized_DataLoader(
    dataset_dirs=DATASET_PATHS,
    transform=data_transforms
 )

def pil_collate(batch):
    curr_imgs, goal_imgs, labels = zip(*batch)   
    labels = torch.stack(labels, 0)           
    return list(curr_imgs), list(goal_imgs), labels
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=pil_collate, pin_memory=False)

# Initializing GroundedSAM2 once
gsam = GroundedSAM2().to(DEVICE)
gsam.eval()

goal_embeds = None
sample_id = 0

with torch.no_grad():
    for curr_imgs, goal_imgs, labels in tqdm(dataloader, desc="Extracting embeddings"):

        labels = labels.to(DEVICE, non_blocking=True)
        curr_embeds = []

        # Compute goal embeddings only once
        if goal_embeds is None:
            goal_features = []
            for cam in range(len(goal_imgs[0])):
                goal_emb, _ = gsam(goal_imgs[0][cam], PROMPT_TXT)
                goal_features.append(goal_emb.squeeze(0).half().cpu())
            goal_embeds = torch.stack(goal_features).unsqueeze(0)  # shape: [1, 4, 256, 64, 64]

        for cam in range(len(curr_imgs[0])):
            curr_emb, _ = gsam(curr_imgs[0][cam], PROMPT_TXT)
            curr_embeds.append(curr_emb.squeeze(0).half().cpu())

        curr_embeds = torch.stack(curr_embeds).unsqueeze(0)  # shape: [1, 4, 256, 64, 64]

        for b in range(curr_embeds.size(0)):
            out_path = os.path.join(FEATURES_DIR, f"{sample_id:05d}.pt")
            torch.save({
                'current_embeddings': curr_embeds[b],
                'goal_embeddings'  : goal_embeds[0],  
                'labels'           : labels[b]
            }, out_path)
            sample_id += 1