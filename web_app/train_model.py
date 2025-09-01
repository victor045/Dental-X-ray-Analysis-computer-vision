#!/usr/bin/env python3
"""
Script to train and save a model for the web application
This script demonstrates how to train a classification model and save it
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Add the src directory to the path
sys.path.append('../src')
from dentexmodel.models.toothmodel_basic import ResNet50Model

class SimpleToothDataset(Dataset):
    """Simple dataset for demonstration purposes"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['Healthy', 'Cavity', 'Crown', 'Filling']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # This is a placeholder - you should implement proper data loading
        # based on your actual dataset structure
        self.samples = []
        
        # Example structure (modify based on your data):
        # for class_name in self.classes:
        #     class_dir = os.path.join(data_dir, class_name)
        #     if os.path.exists(class_dir):
        #         for img_name in os.listdir(class_dir):
        #             if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        #                 self.samples.append((
        #                     os.path.join(class_dir, img_name),
        #                     self.class_to_idx[class_name]
        #                 ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dummy_model():
    """Create a dummy model for demonstration purposes"""
    model = ResNet50Model(n_outputs=4).create_model()
    
    # Initialize with random weights (in practice, you'd train this)
    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
    
    return model

def train_model(model, train_loader, num_epochs=5, device='cpu'):
    """Train the model"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(train_loader):.4f}')

def save_model(model, save_path):
    """Save the trained model"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model state dict
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    """Main function to train and save the model"""
    print("Dental Image Classification Model Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset (you'll need to modify this based on your actual data)
    data_dir = "../data/images"  # Adjust this path to your data directory
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist.")
        print("Creating a dummy model for demonstration purposes...")
        
        # Create dummy model
        model = create_dummy_model()
        model.to(device)
        
        # Save the dummy model
        save_path = "../data/model/classification_model.pth"
        save_model(model, save_path)
        
        print("\nDummy model created and saved!")
        print("To train with real data:")
        print("1. Place your training images in the data directory")
        print("2. Modify the SimpleToothDataset class to load your data")
        print("3. Run this script again")
        
        return
    
    # Create dataset and dataloader
    dataset = SimpleToothDataset(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Create model
    model = ResNet50Model(n_outputs=4).create_model()
    model.to(device)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, num_epochs=5, device=device)
    
    # Save model
    save_path = "../data/model/classification_model.pth"
    save_model(model, save_path)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {save_path}")
    print("\nYou can now run the web application with:")
    print("cd web_app")
    print("python app.py")

if __name__ == "__main__":
    main()

