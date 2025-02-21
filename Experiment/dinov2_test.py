import torch
from torchvision import transforms
from PIL import Image

data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model.eval()
model.cuda()

image = Image.open('/home/ben/spot_IL/dataset_initial/goal/0.png').convert('RGB')
image = data_transforms(image)
image = image.unsqueeze(0)
image = image.to('cuda')

output = model.forward_features(image)

print(output['x_norm_patchtokens'].shape)