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
            # batch_first = True,       
            # bias        = False
        )

    def forward(self, query, key_value):
        # query and key_value: (B, C, H, W)
        B, C, H, W = query.shape     

        print(f"[Cross-Attention] query {query.shape}  key_value {key_value.shape}")            

        # (H*W, B, C)
        query_flat = query.view(B, C, -1).permute(2, 0, 1)
        key_value_flat = key_value.view(B, C, -1).permute(2, 0, 1)
        attn_output, attn_weight = self.mha(query_flat, key_value_flat, key_value_flat)
        print(f"[Cross-Attention] attn_output {attn_output.shape} attn_weights {attn_weight.shape}")
       
       
        # (B, C, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        print(f"[Cross-Attention] out {attn_output.shape}") 
        
        return attn_output, attn_weight
# -------------------------------------------------------------------------------

class GsamMlp5(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        num_trunk_channels = 256
        self.num_cameras = 4 

        self.cross_attention = FlashCrossAttention(embed_dim=num_trunk_channels, num_heads=8)
        
        # Fully connected layers.
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256*4*4*2, 1024),
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


    def forward(self, curr, goal):
        batch_size = curr.size(0)

        # Features from GSAM
        # [GSAM-MLP] curr torch.Size([64, 4, 256, 64, 64])  goal torch.Size([64, 4, 256, 64, 64])
        print(f"[GSAM-MLP] curr {curr.shape}  goal {goal.shape}")
        

        # Stacking 4 current features
        # Stacking 4 goal features 
        current_cat = torch.cat([curr[:, i] for i in range(self.num_cameras)], dim=3)  
        goal_cat    = torch.cat([goal[:, i] for i in range(self.num_cameras)], dim=3) 
        # [GSAM-MLP] current_cat torch.Size([64, 256, 64, 256])  goal_cat torch.Size([64, 256, 64, 256])
        print(f"[GSAM-MLP] current_cat {current_cat.shape}  goal_cat {goal_cat.shape}")

        current_cat = self.reduce(current_cat)  # Reduce the spatial dimensions
        goal_cat = self.reduce(goal_cat)  # Reduce the spatial dimensions


        # Curr - Goal Cross- Attention 
        curr_goal_attenion, cg_attention_score = self.cross_attention(current_cat, goal_cat)
        curr_attended = current_cat + curr_goal_attenion
        # [GSAM-MLP] curr_att torch.Size([64, 256, 32, 128])
        print(f"[GSAM-MLP] curr_att {curr_attended.shape}")

        # Goal - Current Cross- Attention
        goal_curr_attenion, gc_attention_score = self.cross_attention(goal_cat, current_cat)
        goal_attended = goal_cat + goal_curr_attenion
        # [GSAM-MLP] goal_att torch.Size([64, 256, 32, 128]
        print(f"[GSAM-MLP] goal_att {goal_attended.shape}")

        # Average pooling 4x4
        curr_feat = self.global_pool(curr_attended).reshape(batch_size, -1)  
        goal_feat = self.global_pool(goal_attended).reshape(batch_size, -1)     
        # [GSAM-MLP] curr_pool torch.Size([64, 4096])  goal_pool torch.Size([64, 4096])  
        print(f"[GSAM-MLP] curr_pool {curr_feat.shape}  goal_pool {goal_feat.shape}")
        
        # Concatenate current and goal features
        features = torch.cat([curr_feat, goal_feat], dim=1)
        # [GSAM-MLP] concat features torch.Size([64, 8192]
        print(f"[GSAM-MLP] concat features {features.shape}")    

        # MLP Layers
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        # [GSAM-MLP] after FC4 torch.Size([64, 1024])
        print(f"[GSAM-MLP] after FC4 {x.shape}")                            

        output_x = self.fc_layer_x(x)
        output_y = self.fc_layer_y(x)
        output_r = self.fc_layer_r(x)

        outputs = torch.stack([output_x, output_y, output_r], dim=1)
        # [GSAM-MLP] outputs torch.Size([64, 3, 3])
        print(f"[GSAM-MLP] outputs {outputs.shape}")   

        return outputs, cg_attention_score, gc_attention_score

       