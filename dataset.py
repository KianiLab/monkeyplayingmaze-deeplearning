import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torchvision.models as models

from torchvision.transforms.functional import to_tensor

from torch import nn, optim

from torchvision import transforms

import torch.nn.functional as F
import random 


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torch import nn, optim
from torchvision.transforms import ToPILImage, Resize, ToTensor, Lambda




# one hot encoding 

# integer labels 
def direction_to_label(direction):
    mapping = {'down': 0, 'left': 1, 'right': 2, 'up': 3}
    if direction not in mapping:
        print(f"Unexpcted direction:{direction}")
        return None
    return mapping[direction] 


def read_file_and_extract_features_labels(file_path):
    # Skip macOS metadata files that start with '._'
    if file_path.split('/')[-1].startswith('._'):
        print(f"Skipping metadata file: {file_path}")
        return None
    try:
        combined_data = []
        with open(file_path, 'r', encoding='ascii', errors='surrogateescape') as file:

            next(file)  # Skip the header line if present
            for line in file:
                parts = line.strip().split(',')
                # Ensure we have enough parts for prev_x, prev_y, current_x, current_y, next_x, next_y, and direction
                if len(parts) < 7:
                    print(f"Invalid line in file: {file_path}")
                    continue
                try:
                    # Extract features and convert direction to an integer label
                    prev_x, prev_y, current_x, current_y = [int(parts[i]) for i in range(4)]
                    direction = parts[-1].lower().strip()
                    label = direction_to_label(direction)  # Ensure you have a definition for this function
                    if label is not None:
                        combined_data.append(([prev_x, prev_y, current_x, current_y], label))
                    else:
                        print(f"Unexpected direction: '{direction}' in file: {file_path}")
                except ValueError as e:
                    print(f"Error converting line to integers in {file_path}: {e}")
                    continue  # Skip this line and continue with the next

        if not combined_data:
            print(f"No valid data found in file: {file_path}")
            return None

        return combined_data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None







# Custom transformation: Resize and pad images to a fixed size
def resize_and_pad(img, target_size=(2250, 3375), padding_color=(0, 0, 0)):
    # Resize image
    img = TF.resize(img, [int(target_size[0] * img.size[1] / img.size[0]), target_size[1]], interpolation=Image.BILINEAR)
    
    # Calculate padding
    padding = [(target_size[1] - img.size[0]) // 2, 
               (target_size[0] - img.size[1]) // 2,
               (target_size[1] - img.size[0] + 1) // 2,
               (target_size[0] - img.size[1] + 1) // 2]
    
    # Apply padding
    img = TF.pad(img, padding=padding, padding_mode='constant', fill=padding_color)
    return img


  

my_transforms = transforms.Compose([
    transforms.Lambda(resize_and_pad),  # Use the defined function directly
    transforms.ToTensor(),
])


class MonkeyMazeDataset(Dataset):
    def __init__(self, image_dir, trajectory_dir, transform=None):
        self.image_dir=image_dir
        self.trajectory_dir = trajectory_dir
        # self.labels=labels
        # self.transform = transform if transform is not None else resize_and_pad()
        self.transform = my_transforms
        self.pairs=self.findpairs()
       
        print(f" Found {len(self.pairs)} image-feature-label pairs.")

  

    # def is_valid_trajectory(self,file_path):
    #     with open(file_path,'r') as file:
    #         content=file.read().strip()
    #         if not content or "empty" in content:
    #             return False
    #     return True
    
    def is_valid_trajectory(self, file_path):
         try:
            with open(file_path, 'r', encoding='ascii',errors='surrogateescape') as file:
                 content = file.read().strip()
         except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return False
 
         if not content or "empty" in content:
              return False
         return True
    


    def findpairs(self):
        pairs=[]
        # image_files = os.listdir(self.image_dir)
        # trajectory_files = os.listdir(self.trajectory_dir)
        trajectory_data = {} 
        for traj_file in sorted(os.listdir(self.trajectory_dir)):
            if traj_file.endswith('.txt'):
          
                full_path= os.path.join(self.trajectory_dir, traj_file)
                if not self.is_valid_trajectory(full_path):
                    continue
                date = traj_file.split('_')[1]
                trial_num = traj_file.split('trial_')[1].split('_')[0]
                base_name = f"{date}_trial_{trial_num}"
                
               
                data= read_file_and_extract_features_labels(full_path)
                if data:
                    # is not empty 
                    trajectory_data[base_name] = data
#match image files to trajectory data 
        for img_file in sorted(os.listdir(self.image_dir)):
            if img_file.endswith('.png'):
                date = img_file.split('_')[1]
                trial_num = img_file.split('trial_')[1].split('.')[0]
                base_name = f"{date}_trial_{trial_num}"

                if base_name in trajectory_data:
                    for features,label in trajectory_data[base_name]:
                        pairs.append((os.path.join(self.image_dir, img_file), features, label))
        print("sample matche pairs:")
        for pair in pairs[:5]:
            print(pair)
        
        return pairs
    



           
    def __len__(self):
        return len(self.pairs)


    def __getitem__(self,idx):
        img_path, features,label=self.pairs[idx]
     
        image = Image.open(img_path)
        
        
        # # Apply the transformation
        # image = my_transforms(image)
        
        if self.transform:
            image = self.transform(image)

        # convert features coordinates to a tensor
        features_tensor = torch.tensor(features, dtype=torch.float)
         # Apply one-hot encoding to the label index
        # label_one_hot = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=4)
        label_tensor = torch.tensor(label, dtype=torch.long)
        # print(f"Accessing pair #{idx}: Image path - {img_path}, Label - {label}"
     
        return image, features_tensor,label_tensor
            


#     from torchvision import transforms

#     my_transforms = transforms.Compose([
#     transforms.Lambda(resize_and_pad),  # Use the defined function directly
#     transforms.ToTensor(),
# ])



__all__ = ['MonkeyMazeDataset', 'my_transforms']