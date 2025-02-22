import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SPOT_SingleStep_DataLoader(Dataset):
    def __init__(self, dataset_dir, transform=None, cuda=False):
        self.current_images = []
        self.labels = np.empty([0, 3])
        self.goal_image_tags = []

        trajectories = [item for item in os.listdir(dataset_dir) if item.startswith('traj_')]
        trajectories = sorted(trajectories, key=lambda x: int(x[5:]))
        for trajectory in trajectories:
            traj_imgs, traj_labels, traj_goal = self.extract_trajectory(dataset_dir, trajectory, transform)
            self.current_images.extend(traj_imgs)
            self.labels = np.vstack([self.labels, traj_labels])
            self.goal_image_tags.extend(traj_goal)
        
        if not (len(self.current_images) == self.labels.shape[0] == len(self.goal_image_tags)):
            raise ValueError(f"Data length not consistent: "
                            f"current_images={len(self.current_images)}, "
                            f"labels={self.labels.shape[0]}, "
                            f"goal_image_tags={len(self.goal_image_tags)}")
        else:
            self._len = len(self.current_images)
        
        # Load Goal Images
        goal_image_dir = os.path.join(dataset_dir, 'Goal_Images')
        self.goal_images = {}

        for goal_img in sorted(os.listdir(goal_image_dir), key=lambda x: int(x[5:-4])):
            goal_img_path = os.path.join(goal_image_dir, goal_img)
            img = Image.open(goal_img_path)
            if transform:
                    img = transform(img)
            else:
                img = transforms.ToTensor()(img)

            if cuda:
                img = img.cuda()
            img = img.to(dtype=torch.float32)
            self.goal_images[goal_img[:-4]] = img

        # Convert to Tensor
        if cuda:
            self.current_images = torch.stack(self.current_images, dim=0).cuda()
            self.labels = torch.tensor(self.labels).cuda()
        else:
            self.current_images = torch.stack(self.current_images, dim=0)
            self.labels = torch.tensor(self.labels)

        self.current_images = self.current_images.to(dtype=torch.float32)
        self.labels = self.labels.to(dtype=torch.float32)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        goal_img = self.goal_images[self.goal_image_tags[idx]]
        return self.current_images[idx], goal_img, self.labels[idx]
    
    @staticmethod
    def extract_trajectory(dataset_dir, trajectory, transform):

        trajectory_dir = os.path.join(dataset_dir, trajectory)
        steps = sorted(os.listdir(trajectory_dir))[:-1]
        
        # Goal Image Tags
        goal_tags = [trajectory] * len(steps)

        # Label
        label_path = os.path.join(trajectory_dir, 'labels.npy')
        traj_labels = np.load(label_path)

        # Current Images
        traj_imgs = []
        for step in steps:
            step_path = os.path.join(trajectory_dir, step)
            step_imgs = []

            for i in range(5):
                img_path = os.path.join(step_path, f'{i}.jpg')
                img = Image.open(img_path)
                if transform:
                    img = transform(img)
                else:
                    img = transforms.ToTensor()(img)
                img = img.to(dtype=torch.float32)
                step_imgs.append(img)
            
            step_imgs = torch.stack(step_imgs, dim=0)
            traj_imgs.append(step_imgs)

        return traj_imgs, traj_labels, goal_tags
