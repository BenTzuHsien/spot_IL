import torch
import torch.nn as nn
import torch.nn.functional as F
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key_value):
        attn, _ = self.mha(query, key_value, key_value)
        return attn

class GroundingDinoFeatureExtractor(nn.Module):
    def __init__(self, base_model, device):
        super(GroundingDinoFeatureExtractor, self).__init__()
        self.model = base_model
        self.device = device
        self._features = None
        self.hook_handle = self.model.transformer.encoder.layers[-1].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self._features = output

    def forward(self, images, text_prompts):
        images = images.to(self.device)
        _ = self.model(images, captions=text_prompts)
        return self._features

class DINOCrossAttentionMLP(nn.Module):
    def __init__(self, config_file, weight_file, device, num_cameras=5, embed_dim=256):
        super(DINOCrossAttentionMLP, self).__init__()
        self.device = device
        self.num_cameras = num_cameras

        cfg = SLConfig.fromfile(config_file)
        base_model = build_model(cfg)
        checkpoint = torch.load(weight_file, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = { (k[len("module."): ] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        base_model.load_state_dict(state_dict, strict=False)
        base_model.to(device)

        for param in base_model.parameters():
            param.requires_grad = False

        base_model.eval()

        self.feature_extractor = GroundingDinoFeatureExtractor(base_model, device=device)
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads=8)

        self.fc_layer1 = nn.Sequential(
            nn.Linear(2 * embed_dim * num_cameras, 1024),
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

        self.fc_layer5 = nn.Linear(1024, 3)

        self.to(device)

    def forward(self, current_images, goal_images, text_prompts):
        if goal_images.dim() == 4:  # shape [B, C, H, W]
            goal_images = goal_images.unsqueeze(1).expand(-1, self.num_cameras, -1, -1, -1)
        
        current_features_list = []
        goal_features_list = []

        for cam in range(self.num_cameras):
            curr_img = current_images[:, cam, :, :, :]
            goal_img = goal_images[:, cam, :, :, :]
            curr_feat = self.feature_extractor(curr_img, text_prompts)
            goal_feat = self.feature_extractor(goal_img, text_prompts)

            if curr_feat is None or goal_feat is None:
                print(f"[Camera {cam}] Warning: Feature extraction returned None!")
                continue

            curr_attn = curr_feat + self.cross_attention(curr_feat, goal_feat)
            goal_attn = goal_feat + self.cross_attention(goal_feat, curr_feat)
            curr_pool = curr_attn.mean(dim=1)
            goal_pool = goal_attn.mean(dim=1)
            current_features_list.append(curr_pool)
            goal_features_list.append(goal_pool)

        current_features = torch.cat(current_features_list, dim=1)
        goal_features = torch.cat(goal_features_list, dim=1)
        features = torch.cat([current_features, goal_features], dim=1)
        x = self.fc_layer1(features)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        x = self.fc_layer4(x)
        output = self.fc_layer5(x)
        return output