import torch, os
import torch.nn as nn

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import grounding_dino.groundingdino.datasets.transforms as T

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert

GROUNDING_DINO_CONFIG = os.path.expanduser("~/lib/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.expanduser("~/lib/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = os.path.expanduser("~/lib/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")

class GroundedSAM2(nn.Module):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    BOX_THRESHOLD = 0.45
    TEXT_THRESHOLD = 0.4

    def __init__(self):
        super().__init__()

        # Build Grounding DINO
        self.gdino = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT
        )

        # Build SAM-2
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # freeze parameters
        for p in self.gdino.parameters():  
            p.requires_grad = False

        for p in self.sam2_predictor.model.parameters():  
            p.requires_grad = False

    @torch.no_grad() 
    def forward(self, image, prompt, return_mask=False, fully_masked=True):
        """
        Forward Funcion

        Parameters
        ----------
        image : PIL.Image.Image
            The current image.

        prompt : string
            prompt
        return_mask : boolean
            return_mask
        fully_masked : boolean
            fully_mask

        Returns
        -------
        outputs
            model output.
        """

        image_gdino, _ = self.transform(image, None)
        device = next(self.gdino.parameters()).device
        dtype = next(self.gdino.parameters()).dtype
        image_gdino = image_gdino.to(device=device, dtype=dtype)

        # --- Grounding‑DINO ----------------------------------------------------
        boxes, confidences, labels = predict(
            model=self.gdino,
            image=image_gdino,
            caption=prompt,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            device=device
        )
        if boxes.numel() == 0:
            best_boxes = None
        else:
            best_boxes = boxes[confidences.argmax()]

        # --- SAM‑2 image embedding ----------------------------
        self.sam2_predictor.set_image(image)

        # SAM preprocessor upsamples input to 1024×1024.
        # Resulting token_embed shape: [B, C, H, W] = [B, 256, 64, 64], where H = W = 1024 / 16.
        token_embed = self.sam2_predictor.get_image_embedding().to(dtype=dtype)

        if best_boxes is None and fully_masked is True:
            return torch.zeros_like(token_embed), None
        
        elif best_boxes is None and fully_masked is False:
            return token_embed, None

        w, h = image.size
        scale = torch.tensor([w, h, w, h], device=device, dtype=dtype)
        box_xyxy = box_convert(best_boxes.to(device) * scale, in_fmt="cxcywh", out_fmt="xyxy")
        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy,
            multimask_output=False,
        )
        mask_np = masks[0]


        # [GSAM]  raw mask [B, C, H, W] : torch.Size([1, 1, 224, 224])
        m = (torch.from_numpy(mask_np) # [H, W]
                .to(dtype=dtype, device=device)[None, None] # [B, C, H, W]
            )
        
        # [GSAM]  down-sampled mask [B, C, H, W] : torch.Size([1, 1, 64, 64])  
        m = nn.functional.interpolate(m, size=token_embed.shape[-2:], mode="nearest")
        
        # element-wise masking of the feature map, Broadcast single-channel mask across 256 channels: token_embed * m torch.Size([1, 256, 64, 64])
        feats = token_embed * m

        return (feats, mask_np) if return_mask else (feats, None)
    
if __name__ == '__main__':

    from PIL import Image

    image_path = '/root/Object_Centric_Local_Navigation/0.jpg'
    image = Image.open(image_path)
    text = "green chair."
    
    gsam = GroundedSAM2()
    gsam.to("cuda")
    
    feature, _ = gsam(image, text)
    magnitude = torch.norm(feature.squeeze(0).permute(1, 2, 0), dim=-1)

    from torchvision.transforms.functional import to_pil_image
    min_val = magnitude.min()
    max_val = magnitude.max()
    feature_vis = (((magnitude - min_val) / (max_val - min_val + 1e-8)) * 225).to(torch.uint8)
    img = to_pil_image(feature_vis)
    img.save("magnitude_map.png")