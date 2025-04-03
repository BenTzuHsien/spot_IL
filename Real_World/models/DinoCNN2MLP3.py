import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key_value):
        B, C, H, W = query.shape
        query_flat = query.view(B, C, -1).permute(2, 0, 1)
        key_value_flat = key_value.view(B, C, -1).permute(2, 0, 1)
        attn_output, _ = self.mha(query_flat, key_value_flat, key_value_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output

class DinoCnn2MLP3(nn.Module):
    def __init__(self):
        super(DinoCnn2MLP3, self).__init__()
        self.shared_trunk = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        for param in self.shared_trunk.parameters():
            param.requires_grad = False
        self.shared_trunk.eval()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        num_trunk_channels = 384
        self.num_cameras = 5

        # Convolutional heads output 768 channels (384 * 2)
        self.current_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_trunk_channels, num_trunk_channels, kernel_size=3, padding=1),
                nn.Conv2d(num_trunk_channels, num_trunk_channels * 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_cameras)
        ])

        self.goal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_trunk_channels, num_trunk_channels, kernel_size=3, padding=1),
                nn.Conv2d(num_trunk_channels, num_trunk_channels * 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_cameras)
        ])

        # Updated cross-attention block to match the 768-channel output
        self.cross_attention = CrossAttentionBlock(embed_dim=num_trunk_channels * 2, num_heads=8)

        # Update the fc layer input dimension accordingly: 5 cameras × 768 per head × 2 heads = 7680
        self.fc_layer1 = nn.Sequential(
            nn.Linear(7680, 1024),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_layer3 = nn.Linear(1024, 3)

    def forward(self, current_images, goal_image):
        batch_size = current_images.size(0)
        current_features_list = []
        goal_features_list = []

        # Process goal image trunk features
        with torch.no_grad():
            dino_output = self.shared_trunk.forward_features(goal_image)
        goal_trunk_feat = torch.reshape(dino_output['x_norm_patchtokens'], [-1, 16, 16, 384])
        goal_trunk_feat = goal_trunk_feat.permute(0, 3, 1, 2)

        # Process each camera's current image
        for cam_idx in range(self.num_cameras):
            curr = current_images[:, cam_idx, :, :, :]
            with torch.no_grad():
                dino_output = self.shared_trunk.forward_features(curr)
            curr_feat = torch.reshape(dino_output['x_norm_patchtokens'], [-1, 16, 16, 384])
            curr_feat = curr_feat.permute(0, 3, 1, 2)
            curr_feat = self.current_heads[cam_idx](curr_feat)

            goal_feat = self.goal_heads[cam_idx](goal_trunk_feat)

            curr_attended = curr_feat + self.cross_attention(curr_feat, goal_feat)
            goal_attended = goal_feat + self.cross_attention(goal_feat, curr_feat)

            curr_pooled = self.global_pool(curr_attended).view(batch_size, -1)
            goal_pooled = self.global_pool(goal_attended).view(batch_size, -1)

            current_features_list.append(curr_pooled)
            goal_features_list.append(goal_pooled)

        current_features = torch.cat(current_features_list, dim=1)
        goal_features = torch.cat(goal_features_list, dim=1)
        features = torch.cat([current_features, goal_features], dim=1)

        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        output = self.fc_layer3(x)

        return output
