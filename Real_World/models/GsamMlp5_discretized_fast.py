import torch.nn as nn
import os, glob, torch


# -------------------------------------------------------------------------------
# UPDATED CROSS-ATTENTION FUNCTION 
torch.backends.cuda.enable_flash_sdp(True)          
torch.backends.cuda.enable_mem_efficient_sdp(True) 
torch.backends.cuda.enable_math_sdp(False)         


class FlashCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(FlashCrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            batch_first = True,       
            bias        = False
        )

    def forward(self, query, key_value):
        # query and key_value: (B, C, H, W)
        B, C, H, W = query.shape     

        # [Cross-Attention] query torch.Size([8, 256, 64, 64])  key_value torch.Size([8, 256, 64, 64])
        # print(f"[Cross-Attention] query {query.shape}  key_value {key_value.shape}")         
                                  

        # (H*W, B, C)
        query_flat = query.view(B, C, -1).permute(2, 0, 1)
        key_value_flat = key_value.view(B, C, -1).permute(2, 0, 1)
        attn_output, _ = self.mha(query_flat, key_value_flat, key_value_flat)

        # (B, C, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        # [Cross-Attention] out torch.Size([8, 256, 64, 64])
        # print(f"[Cross-Attention] out {attn_output.shape}") 
        
        return attn_output
# -------------------------------------------------------------------------------

class GsamMlp5(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        num_trunk_channels = 256
        self.num_cameras = 5

        # Camera-specific heads for current images
        self.current_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_trunk_channels, num_trunk_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_cameras)
        ])

        # Camera-specific heads for the goal image
        self.goal_head = nn.Sequential(
                nn.Conv2d(num_trunk_channels, num_trunk_channels, kernel_size=1),
                nn.ReLU(inplace=True)
        )

        # Cross-attention block shared across cameras
        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=4)
        
        # Fully connected layers.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256*10, 1024),
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


    def forward(self, curr, goal):
        batch_size = curr.size(0)

        # [GSAM-MLP] curr torch.Size([8, 5, 256, 64, 64])  goal torch.Size([8, 256, 64, 64])
        # print(f"[GSAM-MLP] curr {curr.shape}  goal {goal.shape}")

        
        current_features_list = []
        goal_features_list = []

        goal_feat = self.goal_head(goal)
        # [GSAM-MLP] goal_feat torch.Size([8, 256, 64, 64])
        # print(f"[GSAM-MLP] goal_feat {goal_feat.shape}")

        for cam_idx in range(self.num_cameras):

            curr_feat = self.current_heads[cam_idx](curr[:, cam_idx])
            # [GSAM-MLP] cam0 curr_feat torch.Size([8, 256, 64, 64])
            # print(f"[GSAM-MLP] cam{cam_idx} curr_feat {curr_feat.shape}")

        
            # Applying cross-attention in both directions.
            curr_attended = curr_feat + self.cross_attention(curr_feat, goal_feat)
            goal_attended = goal_feat + self.cross_attention(goal_feat, curr_feat)
            
            # [GSAM-MLP] cam0 curr_att torch.Size([8, 256, 64, 64])  goal_att torch.Size([8, 256, 64, 64])
            # print(f"[GSAM-MLP] cam{cam_idx} curr_att {curr_attended.shape}  goal_att {goal_attended.shape}")

            # Global pooling.
            curr_pooled = self.global_pool(curr_attended).view(batch_size, -1)  
            goal_pooled = self.global_pool(goal_attended).view(batch_size, -1) 
            # [GSAM-MLP] cam0 curr_pool torch.Size([8, 256])  goal_pool torch.Size([8, 256])
            # print(f"[GSAM-MLP] cam{cam_idx} curr_pool {curr_pooled.shape}  goal_pool {goal_pooled.shape}")
            
            current_features_list.append(curr_pooled)
            goal_features_list.append(goal_pooled)
        
         # Concatenating features from all cameras.
        current_features = torch.cat(current_features_list, dim=1)  # (B, 5*256)
        goal_features = torch.cat(goal_features_list, dim=1)        # (B, 5*256)
        features = torch.cat([current_features, goal_features], dim=1)  # (B, 256*10)
        # [GSAM-MLP] concat features torch.Size([8, 2560])
        # print(f"[GSAM-MLP] concat features {features.shape}")    

        # Fully connected layers.
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        # [GSAM-MLP] after FC4 torch.Size([8, 1024])
        # print(f"[GSAM-MLP] after FC4 {x.shape}")                            

        output_x = self.fc_layer_x(x)
        output_y = self.fc_layer_y(x)
        output_r = self.fc_layer_r(x)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        # [GSAM-MLP] outputs torch.Size([8, 3, 3])
        # print(f"[GSAM-MLP] outputs {outputs.shape}")   

        return outputs

       