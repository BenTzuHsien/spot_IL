import torch
import torch.nn as nn
from models.backbones.grounded_sam2 import GroundedSAM2
from models.modules.flash_cross_attention import FlashCrossAttention

class GsamMlp5Uni(nn.Module):
    def __init__(self, use_gsam=True):
        super().__init__()

        if use_gsam:
            self.use_gsam = True
            self.gsam = GroundedSAM2()
        else:
            self.use_gsam = False

        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        num_trunk_channels = 256
        self.num_cameras = 4 

        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=2)
        
        # Fully connected layers.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256*8*8, 1024),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer_x = nn.Linear(1024, 3)
        self.fc_layer_y = nn.Linear(1024, 3)
        self.fc_layer_r = nn.Linear(1024, 3)
        self.reduce = nn.Conv2d(256, 256, kernel_size=2, stride=2)

    def set_goal(self, goal_images, text):
        """
        Funcion to set Goal Condiction

        Parameters
        ----------
        goal_images : list of PIL.Image.Image or torch.Tensor
            The goal images. If `use_gsam` is True, this should be a list of PIL images;
            otherwise, a batch tensor of shape (B, N, C, H, W).
        text : string
            promt

        Returns
        -------
        outputs
            model output.
        """
        if self.use_gsam:
            goal_embeddings = []
            for image in goal_images:
                embedding, _ = self.gsam(image, text)
                goal_embeddings.append(embedding.squeeze(0))

            self.goal_embeddings = torch.stack(goal_embeddings).unsqueeze(0)
        else:
            self.goal_embeddings = goal_images

        self.prompt = text

    def forward(self, current_images):
        """
        Forward Funcion

        Parameters
        ----------
        curr : list of PIL.Image.Image or torch.Tensor
            The current images. If `use_gsam` is True, this should be a list of PIL images;
            otherwise, a batch tensor of shape (B, N, C, H, W).

        goal : list of PIL.Image.Image or torch.Tensor
            The goal images. Same format logic as `curr` depending on `use_gsam`.

        Returns
        -------
        outputs
            model output.
        """
        if self.use_gsam:

            current_embeddings = []
            for image in current_images:
                embedding, _ = self.gsam(image, self.prompt)
                current_embeddings.append(embedding.squeeze(0))

            current_embeddings = torch.stack(current_embeddings).unsqueeze(0)

        else:
            current_embeddings = current_images

        batch_size = current_embeddings.size(0)

        # Features from GSAM
        # [GSAM-MLP] curr torch.Size([64, 4, 256, 64, 64])  goal torch.Size([64, 4, 256, 64, 64])
        # print(f"[GSAM-MLP] curr {current_embeddings.shape}  goal {self.goal_embeddings.shape}")

        # Stacking 4 current features
        # Stacking 4 goal features 
        current_cat = torch.cat([current_embeddings[:, i] for i in range(self.num_cameras)], dim=3)  
        goal_cat    = torch.cat([self.goal_embeddings[:, i] for i in range(self.num_cameras)], dim=3) 
        # [GSAM-MLP] current_cat torch.Size([64, 256, 64, 256])  goal_cat torch.Size([64, 256, 64, 256])
        # print(f"[GSAM-MLP] current_cat {current_cat.shape}  goal_cat {goal_cat.shape}")

        current_cat = self.reduce(current_cat)  # Reduce the spatial dimensions
        goal_cat = self.reduce(goal_cat)  # Reduce the spatial dimensions


        # Cross- Attention 
        curr_goal_attenion, attention_score = self.cross_attention(current_cat, goal_cat)
        curr_attended = current_cat + curr_goal_attenion

        # [GSAM-MLP] curr_att torch.Size([64, 256, 32, 128])
        # print(f"[GSAM-MLP] curr_att {curr_attended.shape}")

        # Average pooling 8x8
        features = self.global_pool(curr_attended).reshape(batch_size, -1)  
        # goal_feat = self.global_pool(goal_cat).reshape(batch_size, -1)       

        # [GSAM-MLP] curr_pool torch.Size([64, 16384])  
        # print(f"[GSAM-MLP] curr_pool {features.shape}")

        # MLP Layers
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        # [GSAM-MLP] after FC4 torch.Size([64, 1024])
        # print(f"[GSAM-MLP] after FC4 {x.shape}")

        output_x = self.fc_layer_x(x)
        output_y = self.fc_layer_y(x)
        output_r = self.fc_layer_r(x)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        # [GSAM-MLP] outputs torch.Size([64, 3, 3])
        # print(f"[GSAM-MLP] outputs {outputs.shape}")

        return outputs, attention_score
    
if __name__ == '__main__':

    import os
    from PIL import Image
    from collections import OrderedDict

    current_image_dir = '/root/Object_Centric_Local_Navigation/14'
    goal_image_dir = '/root/Object_Centric_Local_Navigation/Goal_Images'

    current_images = []
    goal_images = []
    for i in range(4):
        current_image = Image.open(os.path.join(current_image_dir, f'{i}.jpg'))
        current_images.append(current_image)

        goal_image = Image.open(os.path.join(goal_image_dir, f'{i}.jpg'))
        goal_images.append(goal_image)

    model = GsamMlp5Uni(use_gsam=True).to(device="cuda", dtype=torch.float)
    weight_path = '/root/Object_Centric_Local_Navigation/weights/uni_1000.pth'
    state_dict = torch.load(weight_path, map_location='cuda')
    if any(k.startswith("module.") for k in state_dict.keys()):
        # Trained with DataParallel, strip "module."
        new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict, strict=False)
    model.set_goal(goal_images, "green chair.")
    output, _ = model(current_images)
    print(torch.argmax(output, dim=2))