import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SPOT_SingleStep_DataLoader(Dataset):
    def __init__(self, dataset_dirs, transform=None, device='cpu'):
        self.transform = transform
        self.device = device

        self.current_images_paths = []
        self.labels = np.empty([0, 3])
        self.goal_image_paths = []

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for dataset_dir in dataset_dirs:

            trajectories = [item for item in os.listdir(dataset_dir) if item.startswith('traj_')]
            trajectories = sorted(trajectories, key=lambda x: int(x[5:]))
            for trajectory in trajectories:
                traj_imgs_paths, traj_labels, traj_goal = self.extract_trajectory(dataset_dir, trajectory)
                self.current_images_paths.extend(traj_imgs_paths)
                self.labels = np.vstack([self.labels, traj_labels])
                self.goal_image_paths.extend(traj_goal)

        if not (len(self.current_images_paths) == self.labels.shape[0] == len(self.goal_image_paths)):
            raise ValueError(f"Data length not consistent: "
                            f"current_images={len(self.current_images_paths)}, "
                            f"labels={self.labels.shape[0]}, "
                            f"goal_image_tags={len(self.goal_image_tags)}")
        else:
            self._len = len(self.current_images_paths)
        

        self.labels = torch.tensor(self.labels).to(device)
        self.labels = self.labels.to(dtype=torch.float32)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        step_imgs = self.load_step_images(self.current_images_paths[idx])
        goal_img = self.load_goal_images(self.goal_image_paths[idx])
        return step_imgs, goal_img, self.labels[idx]
    
    @staticmethod
    def extract_trajectory(dataset_dir, trajectory):

        trajectory_dir = os.path.join(dataset_dir, trajectory)
        steps = sorted(os.listdir(trajectory_dir))[:-1]
        
        # Goal Image Tags
        goal_image_path = os.path.join(dataset_dir, 'Goal_Images', f'{trajectory}.jpg')
        goal_image_paths = [goal_image_path] * len(steps)

        # Label
        label_path = os.path.join(trajectory_dir, 'labels.npy')
        traj_labels = np.load(label_path)

        # Current Images
        traj_imgs_paths = []
        for step in steps:
            step_path = os.path.join(trajectory_dir, step)
            step_imgs_paths = []

            for i in range(5):
                img_path = os.path.join(step_path, f'{i}.jpg')
                # img = Image.open(img_path)
                # if transform:
                #     img = transform(img)
                # else:
                #     img = transforms.ToTensor()(img)
                # img = img.to(dtype=torch.float32)
                step_imgs_paths.append(img_path)
            
            # step_imgs = torch.stack(step_imgs, dim=0)
            traj_imgs_paths.append(step_imgs_paths)

        return traj_imgs_paths, traj_labels, goal_image_paths

    def load_step_images(self, step_image_paths):
        step_imgs = []
        for img_path in step_image_paths:
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            img = img.to(dtype=torch.float32)
            step_imgs.append(img)
        
        step_imgs = torch.stack(step_imgs, dim=0).to(dtype=torch.float32).to(self.device)

        return step_imgs

    def load_goal_images(self, goal_image_path):
        img = Image.open(goal_image_path)
        if self.transform:
                img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        img = img.to(dtype=torch.float32).to(self.device)

        return img
