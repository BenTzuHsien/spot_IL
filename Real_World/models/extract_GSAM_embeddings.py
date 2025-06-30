import os, torch, glob
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from SPOT_SingleStep_Discredtized_DataLoader import SPOT_SingleStep_Discretized_DataLoader
import cv2
import numpy as np

import torch.nn as nn
from sam2.build_sam import build_sam2
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T
import torchvision.transforms.functional as F

# GSAM IMPLEMENTATION
# ----------------------------------------------------------------------------
# Configuration paths
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# Thresholds
BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.4

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_batch_for_gdino(batch):

    # doing what load_image(image_path) for GDino did

    device = batch.device
    B, C, H, W = batch.shape

    size = 800
    max_size = 1333

    scale = size / min(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_h = int(round(new_h * scale))
        new_w = int(round(new_w * scale))

    imgs = F.resize(batch, [new_h, new_w], antialias=True)
    imgs = F.normalize(imgs, IMAGENET_MEAN, IMAGENET_STD)
    return imgs


def preprocess_batch_for_sam2(batch):
    # SAM API is written for ordinary uint8 RGB image [H, W, C]
    batch_out = (batch.clamp(0, 1) * 255).to(torch.uint8)
    batch_out = batch_out.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [img for img in batch_out]


class GSAM(nn.Module):

    def __init__(self):
        super().__init__()

        # Build Grounding DINO
        self.gdino = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT
            # device='cpu'
        )

        # Build SAM-2
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # freeze parameters 
        for p in self.sam2_predictor.model.parameters():  
            p.requires_grad = False                       
        
        # eval mode
        self.sam2_predictor.model.eval()
        self.gdino.eval()

    
    @torch.no_grad() 
    def forward(self, image, prompt, return_mask =True):

        # pass the default return_mask=False during training !!!
        B = image.size(0)

        # converting image tensors to correct formats :
        gdino_imgs = preprocess_batch_for_gdino(image)  
        sam2_imgs = preprocess_batch_for_sam2(image)    

        # --- Grounding‑DINO ----------------------------------------------------                                
        best_boxes = []
        
        for i in range(B):     
            boxes, confidences, labels = predict(
                model=self.gdino,
                image=gdino_imgs[i],
                caption=prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
                # device=DEVICE
            )
            if boxes.numel() == 0:
                best_boxes.append(None)
            else:
                best_boxes.append(boxes[confidences.argmax()])  


        # --- SAM‑2 image embedding ----------------------------
        feats = []
        masks_out = []

        for i in range(B):
            self.sam2_predictor.set_image(sam2_imgs[i])

            # token_embed size : (Batch, C, H, W)
            # SAM pre-processing will upsample images to [HxW] of [1024x1024] 
            # H = W = 1024 / 16 = 64
            # C = 256
            token_embed = self.sam2_predictor.get_image_embedding()
            # [GSAM] token_embed  [B, C, H, W]: torch.Size([1, 256, 64, 64])
            # print(f"[GSAM] token_embed  [B, C, H, W]: {token_embed.shape}") 

            if best_boxes[i] is None:
                feats.append(token_embed)
                masks_out.append(None)
                continue

            h, w, _ = sam2_imgs[i].shape
            # scale = torch.tensor([w, h, w, h], device=DEVICE, dtype=best_boxes[i].dtype)
            scale = torch.tensor([w, h, w, h], device=image.device, dtype=best_boxes[i].dtype)
            box_xyxy = box_convert(best_boxes[i][None].to(image.device) * scale, in_fmt="cxcywh", out_fmt="xyxy")[0]
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy.unsqueeze(0).cpu().numpy(),
                multimask_output=False,
            )

            mask_np = masks[0]

            # [GSAM]  raw mask [B, C, H, W] : torch.Size([1, 1, 224, 224])
            m = (
                    torch.from_numpy(mask_np) # [H, W]
                    .float()
                    .to(token_embed.device)[None, None] # [B, C, H, W]
                )  
            # print(f"[GSAM]  raw mask [B, C, H, W] : {m.shape}")
            
            m = nn.functional.interpolate(m, size=token_embed.shape[-2:], mode="nearest")
            # [GSAM]  down-sampled mask [B, C, H, W] : torch.Size([1, 1, 64, 64])  
            # print(f"[GSAM] down-sampled mask [B, C, H, W] : {m.shape}")
            

            feats.append(token_embed * m)
            # element-wise masking of the feature map 
            # [GSAM] token_embed * m torch.Size([1, 256, 64, 64])
            # broadcasting the single-channel mask across the 256 feature channels.
            # print(f"[GSAM] token_embed * m {(token_embed*m).shape}")  

            masks_out.append(mask_np)

        # we return masks for visualization only if needed
        feat_batch = torch.cat(feats, dim=0)  # (B,256,64,64)
        return (feat_batch, masks_out) if return_mask else (feat_batch, None)

# ----------------------------------------------------------------------------
# PATH AND PARAMETER DEFINITIONS 
# ----------------------------------------------------------------------------
# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Setup Destination
DATASET_NAMES = ['map01_01a', 'map01_01b', 'map01_02a', 'map01_02b', 'map01_03a', 'map01_03b']
DATASET_DIR = '/data/shared_data/SPOT_Real_World_Dataset/cleanup_dataset/'

FEATURES_DIR = os.path.join(os.getcwd(), 'gsam_features') 
VIS_DIR = os.path.join(os.getcwd(), 'visualizations') 

# Text prompt
TEXT_PROMPT = "green chair." 

# Hyper Parameters
BATCH_SIZE = 1
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------- visualization functions -------------------------
def to_bgr(img_tensor):
    # converting tensor to uint8 BGR for OpenCV
    img = (img_tensor.clamp(0, 1)
                     .cpu()
                     .numpy()
                     .transpose(1, 2, 0) * 255).astype(np.uint8)
    return img[..., ::-1]          


def overlay_mask(image_bgr, mask_bool, alpha=0.45):   
    # semi-transparent red mask over the segmented region
    overlay = image_bgr.copy()
    overlay[mask_bool] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, image_bgr, 1-alpha, 0)

def show_mask(image_bgr, mask_bool): 
    # only segmented region w/ no background              
    mask = np.zeros_like(image_bgr)
    mask[mask_bool] = image_bgr[mask_bool]
    return mask
# ----------------------------------------------------

# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
        # Setup Dataset Path
        DATASET_PATHS = []
        for dataset_name in DATASET_NAMES:
            dataset_path = os.path.join(DATASET_DIR, f'{dataset_name}')
            if not os.path.exists(dataset_path):
                print(f'Dataset {dataset_name} does not exist!')
                exit()
            DATASET_PATHS.append(dataset_path)


        os.makedirs(FEATURES_DIR, exist_ok=True)
        os.makedirs(VIS_DIR,  exist_ok=True)

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),               
        ])

        train_dataset = SPOT_SingleStep_Discretized_DataLoader(
            dataset_dirs=DATASET_PATHS,
            transform=data_transforms
        )
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

        trunk = GSAM().to(DEVICE).eval()  

        sample_id = 0
        vis_id = 0
        for current_images, goal_image, labels in tqdm(train_dataloader, desc="Extracting embeddings"):
            # current_images: (1, 5, 3, 224, 224)  torch.float32  cuda:0
            # goal_image: (1, 3, 224, 224)  torch.float32  cuda:0

            current_images = current_images.to(DEVICE)
            goal_image = goal_image.to(DEVICE)
            labels = labels.to(DEVICE)

            
            goal_embed, goal_masks = trunk(goal_image, TEXT_PROMPT, return_mask=True)  
            #  goal_embed (GPU): (1, 256, 64, 64)  torch.float32  cuda:0
            
            # shrinking to 16 bit precision
            # goal_embed (CPU half): (1, 256, 64, 64)  torch.float16  cpu
            goal_embed = goal_embed.half().cpu() 
                        
            curr_embeds = []
            cam_masks=[]

            for cam in range(5):
                curr_emb, curr_masks = trunk(current_images[:, cam], TEXT_PROMPT, return_mask=True)
                # curr_emb cam0: (1, 256, 64, 64)  torch.float32  cuda:0
                curr_embeds.append(curr_emb.half().cpu())
                cam_masks.append(curr_masks)
            
            curr_embeds = torch.stack(curr_embeds, dim=1)       
            #  curr_embeds final: (1, 5, 256, 64, 64)  torch.float16  cpu

            for b in range(goal_embed.size(0)):
                out_path = os.path.join(FEATURES_DIR, f"{sample_id:06d}.pt")
                torch.save({
                    'current_embeddings': curr_embeds[b],
                    'goal_embeddings'   : goal_embed[b],
                    'labels'        : labels[b]
                }, out_path)
                sample_id += 1
            # ----------------------------------------------
            # Visualization of produced segmentation masks
            if vis_id < 15 :
                img_bgrs = [to_bgr(current_images[0,cam]) for cam in range(5)]  
                top, mid, bot = [], [], []
                for cam in range(5):
                    sam_mask = cam_masks[cam][0]

                    if sam_mask is None:                 
                        sam_mask = torch.zeros((224,224), dtype=torch.bool)
                    else:
                        sam_mask = torch.from_numpy(sam_mask).bool()

                    top.append(show_mask(img_bgrs[cam], sam_mask))
                    mid.append(overlay_mask(img_bgrs[cam], sam_mask))
                    bot.append(img_bgrs[cam])

                grid = cv2.vconcat([cv2.hconcat(top), cv2.hconcat(mid), cv2.hconcat(bot)])
                cv2.imwrite(os.path.join(VIS_DIR, f"vis_{vis_id:06d}.jpg"), grid); 
                vis_id+=1
            # --------------------------------------------------------------------------

if __name__ == '__main__':
    main()







