import torch
import torch.nn as nn
from torchvision.models import resnet18

class FiveResNet18MLP5(nn.Module):
    def __init__(self):
        super(FiveResNet18MLP5, self).__init__()

        # Current Images Set
        # ResNet1
        self.current_resnet1 = resnet18(pretrained=False)
        self.current_resnet1 = nn.Sequential(*list(self.current_resnet1.children())[:-1])
        # ResNet2
        self.current_resnet2 = resnet18(pretrained=False)
        self.current_resnet2 = nn.Sequential(*list(self.current_resnet2.children())[:-1])
        # ResNet3
        self.current_resnet3 = resnet18(pretrained=False)
        self.current_resnet3 = nn.Sequential(*list(self.current_resnet3.children())[:-1])
        # ResNet4
        self.current_resnet4 = resnet18(pretrained=False)
        self.current_resnet4 = nn.Sequential(*list(self.current_resnet4.children())[:-1])
        # ResNet5
        self.current_resnet5 = resnet18(pretrained=False)
        self.current_resnet5 = nn.Sequential(*list(self.current_resnet5.children())[:-1])

        # Goal Images Set
        # ResNet1
        self.goal_resnet1 = resnet18(pretrained=False)
        self.goal_resnet1 = nn.Sequential(*list(self.goal_resnet1.children())[:-1])
        # ResNet2
        self.goal_resnet2 = resnet18(pretrained=False)
        self.goal_resnet2 = nn.Sequential(*list(self.goal_resnet2.children())[:-1])
        # ResNet3
        self.goal_resnet3 = resnet18(pretrained=False)
        self.goal_resnet3 = nn.Sequential(*list(self.goal_resnet3.children())[:-1])
        # ResNet4
        self.goal_resnet4 = resnet18(pretrained=False)
        self.goal_resnet4 = nn.Sequential(*list(self.goal_resnet4.children())[:-1])
        # ResNet5
        self.goal_resnet5 = resnet18(pretrained=False)
        self.goal_resnet5 = nn.Sequential(*list(self.goal_resnet5.children())[:-1])

        # MLP Layers
        self.fc_layer1 = nn.Sequential(
            nn.Linear(5120, 1024),
            nn.ReLU())
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc_layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc_layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc_layer5 = nn.Linear(1024, 1)
    
    def forward(self, current_images, goal_images):
        
        # Forward pass through ResNet
        current_embedding1 = self.current_resnet1(current_images[:, 0, :, :])
        current_embedding1 = torch.flatten(current_embedding1, start_dim=1)
        current_embedding2 = self.current_resnet2(current_images[:, 1, :, :])
        current_embedding2 = torch.flatten(current_embedding2, start_dim=1)
        current_embedding3 = self.current_resnet3(current_images[:, 2, :, :])
        current_embedding3 = torch.flatten(current_embedding3, start_dim=1)
        current_embedding4 = self.current_resnet4(current_images[:, 3, :, :])
        current_embedding4 = torch.flatten(current_embedding4, start_dim=1)
        current_embedding5 = self.current_resnet5(current_images[:, 4, :, :])
        current_embedding5 = torch.flatten(current_embedding5, start_dim=1)

        # Forward pass through ResNet
        goal_embedding1 = self.goal_resnet1(goal_images[:, 0, :, :])
        goal_embedding1 = torch.flatten(goal_embedding1, start_dim=1)
        goal_embedding2 = self.goal_resnet2(goal_images[:, 1, :, :])
        goal_embedding2 = torch.flatten(goal_embedding2, start_dim=1)
        goal_embedding3 = self.goal_resnet3(goal_images[:, 2, :, :])
        goal_embedding3 = torch.flatten(goal_embedding3, start_dim=1)
        goal_embedding4 = self.goal_resnet4(goal_images[:, 3, :, :])
        goal_embedding4 = torch.flatten(goal_embedding4, start_dim=1)
        goal_embedding5 = self.goal_resnet5(goal_images[:, 4, :, :])
        goal_embedding5 = torch.flatten(goal_embedding5, start_dim=1)

        # Concatenate the features
        current_features = torch.cat((current_embedding1, current_embedding2, current_embedding3, current_embedding4, current_embedding5), dim=1)
        goal_features = torch.cat((goal_embedding1, goal_embedding2, goal_embedding3, goal_embedding4, goal_embedding5), dim=1)
        features = torch.cat([current_features, goal_features], dim=1)

        # Forward pass through the fully connected layers
        output1 = self.fc_layer1(features)
        output2 = self.fc_layer2(output1)
        output3 = self.fc_layer3(output2)
        output4 = self.fc_layer4(output3)
        output = self.fc_layer5(output4)
        
        return output