#!/usr/bin/env python
# coding: utf-8

# In[69]:


from kdata import MonkeyMazeDataset, my_transforms
# from dataset import MonkeyMazeDataset,transform_pipeline
import random
import wandb
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
# from memory_profiler import profile
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn 
from sklearn import metrics
import logging
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torchvision.models as models

from torchvision.transforms.functional import to_tensor

from torch import nn, optim



import torch.nn.functional as F
import random 


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torch import nn, optim
from torchvision.transforms import ToPILImage, Resize, ToTensor, Lambda


# In[ ]:





# In[ ]:





# In[70]:


# # Define your image and trajectory directories

# image_dir = '/Users/xinglanzhao/Desktop/monkeymaze/smallimage'
# trajectory_dir = '/Users/xinglanzhao/Desktop/monkeymaze/smalltrajectory'


image_dir = '/scratch/xz3761/DataMonkeymaze/MazeImage'
trajectory_dir = '/scratch/xz3761/DataMonkeymaze/Monkeytrajectory'
dataset = MonkeyMazeDataset(image_dir=image_dir, trajectory_dir=trajectory_dir,transform=my_transforms)

all_features = torch.cat([torch.tensor(pair[1], dtype=torch.float32) for pair in dataset.pairs])
feature_means = all_features.mean(dim=0)
feature_stds = all_features.std(dim=0) 


       

dataset = MonkeyMazeDataset(image_dir=image_dir, trajectory_dir=trajectory_dir,transform=my_transforms,feature_means=feature_means,feature_stds=feature_stds)

 




#  # Debug Manually access a few items from the dataset
# for idx in range(5):  # Adjust this range as needed
    
#     image, features_tensor, label_one_hot = dataset[idx]
#     print(f"Sample {idx} - Features: {features_tensor}, One-hot label: {label_one_hot}")

   
#     plt.imshow(image)
    


# In[71]:


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






# In[72]:


# Set environment variables for W&B
os.environ["WANDB_HTTP_TIMEOUT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "600"
os.environ["WANDB_DEBUG"] = "true" 


# In[73]:


# Define the custom model class customalexnet Load the pre-trained AlexNet model
import torch.nn as nn
import torchvision.models as models
from torchvision.models import alexnet, AlexNet_Weights


class CustomAlexNet(nn.Module):
    def __init__(self, num_traj_features, num_classes):
        super(CustomAlexNet, self).__init__()
        #self.alexnet = models.alexnet(pretrained=True)
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
        
        # freeze the feature layers (weight and biases of the convoloution layer
        for param in self.alexnet.features.parameters():
            param.requires_grad = False 

        
        self.dropout = nn.Dropout(0.5)
        # extend the first classifer layer to accept the trajectory feature prev_x, pre_y, cur_x,cur_y
   
        # Get the number of input features to the first Linear layer in the classifier section
        num_ftrs = self.alexnet.classifier[1].in_features
        self.new_features = nn.Linear(num_ftrs + num_traj_features, 4096)
        
          # Redefining the classifier
        self.alexnet.classifier = nn.Sequential(
            self.new_features,  # new first layer accepting trajectory features
            nn.ReLU(),
            self.dropout,  # apply dropout after ReLU
            nn.Linear(4096, num_classes)  # Output layer adjustment directly to num_classes
        )
       
   
    # call fowrard in trainning loop with like model(img_tensors, extra_features_tensors)
    def forward(self, images, traj_features):
    # Process images through the conv layers to get their feature representations
        img_features = self.alexnet.features(images)
        img_features = self.alexnet.avgpool(img_features)
        img_features = torch.flatten(img_features, 1)
    
    # Combine image features with trajectory features
        combined_features = torch.cat((img_features, traj_features), dim=1)

         # Apply dropout
        combined_features = self.dropout(combined_features)
        #output = self.batch_norm_class(output)  # Apply batch norm
    
    # Pass the combined features through the modified classifier
        output = self.alexnet.classifier(combined_features)
        return output








        





    
        
    


# In[74]:


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger 



# create the log file Path
# log_file_path=os.path.join("/Users/xinglanzhao/Desktop/monkeymaze", 'all_trail_exp.log')
log_file_path=os.path.join("/home/xz3761/reports",'all_trail_exp.log')







# In[76]:


import torch
from tqdm import tqdm





def load_model(model_path, model, optimizer=None, scheduler=None):
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the optimizer state, if optimizer is provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load the scheduler state, if scheduler is provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Optionally, return any other necessary information like epoch, loss, etc.
    epoch = checkpoint['epoch']
   

    return model, optimizer, scheduler, epoch




def evaluate_model(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, traj_features, labels in loader:
            images = images.to(device)
            traj_features = traj_features.to(device)
            labels = labels.to(device)
            scores = model(images, traj_features)
            loss = criterion(scores, labels)
            total_loss += loss.item()
            _, predicted = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(loader), correct / total

def run_training_loop(start, config, model, train_loader, val_loader,device, optimizer, criterion, scheduler,logger):
    num_epochs = config['epochs']
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf') 
    train_acc_history=[]
    val_acc_history=[]
    improvement_threshold=0.001 
    logger.info("Starting training!")
    for epoch in range(start,num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, traj_features, labels) in enumerate(train_loader):
            images = images.to(device)
            traj_features = traj_features.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            scores = model(images, traj_features)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
 
            # Logging training loss to W&B per batch level 
            wandb.log({"train_loss_batch": loss.item(), "step": epoch * len(train_loader) + batch_idx})

        train_loss, train_accuracy = evaluate_model(train_loader, model, criterion, device)
        val_loss, val_accuracy = evaluate_model(val_loader, model, criterion, device)
        scheduler.step(val_loss)


        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

        logger.info(f'Epoch {epoch+1}, curr Train Loss: {train_loss:.4f}, Curr Train Acc: {train_accuracy:.4f}, Curr Val Loss: {val_loss:.4f}, Curr Val Acc: {val_accuracy:.4f}')
        model_path = "/home/xz3761/reports"
        save_path=os.path.join(model_path,"cnn15.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        
                }
    
        torch.save(checkpoint, save_path)
       
              # Save to W&B directory
        wandb_save_path = os.path.join(wandb.run.dir, "model_best12.pth")
        torch.save(checkpoint, wandb_save_path)
        wandb.save(wandb_save_path)
        
          

        # Logging epoc metrics to W & B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
             "learning_rate": scheduler.get_last_lr()[0]  # Log the current learning rate
    
        })
        
    logger.info("Training completed")
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history
    
 



  


def plot_training_results(train_loss_history, val_loss_history, train_acc_history, val_acc_history, model_path):
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Train Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save the plot locally
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        plots_path = os.path.join(model_path, 'training_validation_lossaccplots.png')
        plt.savefig(plots_path, dpi=300)
        plt.close()
        
        # Log the plot image to wandb
        wandb.log({"Training and Validation Plots": wandb.Image(plots_path)})

    except Exception as e:
        print(f"Failed to save and log plots: {e}")



import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import wandb



def DrawConfusionMatrix(model, model_path, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
   

    # Properly load the checkpoint
    checkpoint_path = os.path.join(model_path, "cnn15.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack the batch according to the structure
            data = batch[0]
            traj_features = batch[1]
            labels = batch[2]

            # Move to device
            data, traj_features, labels = data.to(device), traj_features.to(device), labels.to(device)
            outputs = model(data, traj_features)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())  # Move to CPU and convert to numpy array
            ground_truths.extend(labels.cpu().numpy())   # Move to CPU and convert to numpy array
    
    # Ensure ground_truths and predictions are 1-dimensional arrays
    ground_truths = np.array(ground_truths).flatten()
    predictions = np.array(predictions).flatten()
    #   # Debugging information
    # print(f"Ground truths: {ground_truths}")
    # print(f"Predictions: {predictions}")
    
    # Generate confusion matrix
    labels = ['down', 'left', 'right', 'up']
    cm = confusion_matrix(y_true=ground_truths, y_pred=predictions, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plotting
    plt.style.use('default')
    disp.plot()
    plt.title('Confusion Matrix')

    # Save the plot to a file locally
    local_save_path = os.path.join(model_path, 'confusion_matrix.png')
    plt.savefig(local_save_path, dpi=1000)
    plt.show()

    # Save the plot to W&B
    wandb_save_path = os.path.join(wandb.run.dir, 'confusion_matrix.png')
    plt.savefig(wandb_save_path, dpi=1000)
    wandb.save(wandb_save_path)

    # Print the classification report locally
    report = classification_report(ground_truths, predictions, target_names=labels)
    report_path = os.path.join(model_path, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Save the classification report to W&B
    wandb_report_path = os.path.join(wandb.run.dir, 'classification_report.txt')
    with open(wandb_report_path, 'w') as f:
        f.write(report)
    wandb.save(wandb_report_path)








# Increase W&B initialization timeout
def initialize_wandb(config,mode= "online"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            wandb.init(
                project=config['project_name'],
                resume="must",
                config=config,
                id="3ovagcrv", # just example keyid 
                settings=wandb.Settings(start_method="fork", init_timeout=600)
            )
            return True
        except wandb.errors.CommError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)  # wait for 10 seconds before retrying
            else:
                print("Max retries reached. Initialization failed.")
                return False




def main():

    logger = get_logger(log_file_path)
    logger.info("Logger initiazlied.")
    # import multiprocessing as mp
    seed_value = 42  
    set_seed(seed_value) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # model_path = "/Users/xinglanzhao/Desktop/monkeymaze/reports"
    model_path="/home/xz3761/reports" 

    config = {
        "project_name": "MonkeyMaze",
        "epochs": 1000,
        "batch_size": 32,
        "learning_rate": 0.0008,
        'patience': 100000
        
    }

   
    if initialize_wandb(config, mode="online"):
        print("W&B initialized in online mode.")
    elif initialize_wandb(config, mode="offline"):
        print("W&B initialized in offline mode.")
    else:
        print("W&B initialization failed in both online and offline modes. Exiting.")
        return
    
    total_size=len(dataset)
    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15
    train_size = int(train_ratio* total_size)
    val_size= int(val_ratio* total_size)
    test_size = total_size- train_size-val_size
    train_dataset, val_dataset,test_dataset = random_split(dataset, [train_size,val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,num_workers=8,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,num_workers=4, pin_memory=True)


    # Count the number of samples in each class in the training set
    label_counts = Counter()
    for _,_, labels in DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False):
        label_counts.update(labels.numpy())

    total_samples = sum(label_counts.values())
   
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count} samples ({count/total_samples * 100:.2f}%)")
    # Calculate the class weights
    class_freq_train = {cls: count / total_samples for cls, count in label_counts.items()}
    class_weights = {cls: 1.0 / freq for cls, freq in class_freq_train.items()}

# Convert to tensor
  
    class_weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(class_weights.keys())], dtype=torch.float).to(device)

# Optionally normalize the weights
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()
    print("Class weights tensor:", class_weights_tensor)
    
   
        # num_traj_features = 4  # prev_x, prev_y, current_x, current_y
        # num_classes = 4        # up, down, left, right
    model = CustomAlexNet(num_traj_features=4, num_classes=4).to(device)
    
    model_path = "/home/xz3761/reports/cnn14.pth"
  

    # Set up the loss function with the class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    model, optimizer, scheduler, start_epoch = load_model(model_path, model, optimizer, scheduler)
    print(f"Resuming from epoch {start_epoch}")
    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer,factor=0.75, patience=58)
   
    history=run_training_loop(start_epoch+1, config, model, train_loader, val_loader, device, optimizer, criterion,scheduler,logger)
       
    test_loss, test_accuracy = evaluate_model(test_loader, model, criterion, device)
    logger.info(f"Final Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%") 

        # Logging test accuracy to W&B
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
        
     
    DrawConfusionMatrix(model, model_path, test_loader)
        # unpack the variable  and call to plot it 
    plot_training_results(*history, model_path)   
    wandb.finish()

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
