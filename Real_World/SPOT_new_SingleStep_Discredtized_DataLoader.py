import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

class SPOT_SingleStep_Discretized_DataLoader(Dataset):
    def __init__(self, dataset_dirs, transform=None):
        self.transform = transform

        self.current_images_paths = []
        self.labels = np.empty([0, 3])
        self.goal_image_paths = []

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for dataset_dir in dataset_dirs:
            # now folders are named "000", "001"
            trajectories = [item for item in os.listdir(dataset_dir) if item.isdigit()]
            trajectories = sorted(trajectories, key=lambda x: int(x))
            for trajectory in trajectories:
                traj_imgs_paths, traj_labels, traj_goal = self.extract_trajectory(dataset_dir, trajectory)
                # print(f"[DEBUG init] traj {trajectory}:")
                # print(f"steps loaded = {len(traj_imgs_paths)}")
                # print(f"labels shape  = {traj_labels.shape}")
                # print(f"goal entries  = {len(traj_goal)}")
                self.current_images_paths.extend(traj_imgs_paths)
                self.labels = np.vstack([self.labels, traj_labels])
                self.goal_image_paths.extend(traj_goal)

        if not (len(self.current_images_paths) == self.labels.shape[0] == len(self.goal_image_paths)):
            raise ValueError(f"Data length not consistent: "
                            f"current_images={len(self.current_images_paths)}, "
                            f"labels={self.labels.shape[0]}, "
                            f"goal_image_tags={len(self.goal_image_paths)}")
        else:
            self._len = len(self.current_images_paths)
        

        self.labels = torch.tensor(self.labels).to(dtype=torch.long)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        step_imgs = self.load_step_images(self.current_images_paths[idx])
        goal_imgs = self.load_goal_images(self.goal_image_paths[idx])
        label    = self.labels[idx]

        # print(f"[DEBUG getitem] idx={idx}")
        # print(f"  step_imgs.shape = {step_imgs.shape}")
        # print(f"  goal_imgs.shape = {goal_imgs.shape}")
        # print(f"  label           = {label}")

        return step_imgs, goal_imgs, self.labels[idx]
    
        
    
    @staticmethod
    def extract_trajectory(dataset_dir, trajectory):

        trajectory_dir = os.path.join(dataset_dir, trajectory)
        steps = [x for x in os.listdir(trajectory_dir) if x.isdigit()]
        steps = sorted(steps)

        # print(f"[DEBUG extract] traj {trajectory} dir={trajectory_dir}")
        # print(f"   found steps: {steps}")
        
        # Goal Image Tags
        goal_dir = os.path.join(dataset_dir, 'Goal_Image')
        goal_imgs = [os.path.join(goal_dir, f'{i}.jpg') for i in range(4)]
        goal_image_paths = [goal_imgs] * len(steps)

        # Label (was .npy, now CSV) !!!
        label_path = os.path.join(trajectory_dir, 'actions.csv')
        traj_labels = np.loadtxt(label_path, delimiter=' ').reshape(-1, 3)
        # print(f"[DEBUG] actions.csv for traj {trajectory} →\n{traj_labels}")

        # Current Images
        traj_imgs_paths = []
        for step in steps:
            step_path = os.path.join(trajectory_dir, step)
            step_imgs_paths = []

            for i in range(4):
                img_path = os.path.join(step_path, f'{i}.jpg')
                step_imgs_paths.append(img_path)
            
            # step_imgs = torch.stack(step_imgs, dim=0)
            traj_imgs_paths.append(step_imgs_paths)

        # print(f"[DEBUG extract] traj {trajectory} total step-img lists = {len(traj_imgs_paths)}")
        # print(f"[DEBUG extract] traj {trajectory} goal-paths per step = {len(goal_image_paths)}\n")

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
        
        step_imgs = torch.stack(step_imgs, dim=0).to(dtype=torch.float32)
        # print(f"[DEBUG load_step] stacked step_imgs → {step_imgs.shape}")
        return step_imgs

    def load_goal_images(self, goal_image_path):
        imgs = []
        for img_path in goal_image_path:
            img = Image.open(img_path)
            if self.transform:
                    img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            imgs.append(img.to(dtype=torch.float32))
        img = img.to(dtype=torch.float32)
        # print(f"[DEBUG load_goal] stacked goal_imgs → {torch.stack(imgs, dim=0).shape}")
        return torch.stack(imgs, dim=0)