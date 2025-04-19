import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
from matplotlib import pyplot as plt
import wandb
import yaml

from A_Q1 import CNN

# Defining the Transform to resize Input to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize all images to 224x224
    transforms.ToTensor(),                    # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean/std
                        std=[0.229, 0.224, 0.225])
])


# Loading the dataset
dataset_train_path = "inaturalist_12K/train"
dataset_train = datasets.ImageFolder(root=dataset_train_path, transform=transform)
dataset_test_path = "inaturalist_12K/val"
dataset_test = datasets.ImageFolder(root=dataset_test_path, transform=transform)


# Split into training and validation sets
val_split = 0.2
val_size = int(len(dataset_train) * val_split)
train_size = len(dataset_train) - val_size

train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset,shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=50, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the model with the best parameters
model = CNN(filters=[32,32,32,32,32],kernel_sizes=[5,5,5,5,5],pool_sizes = [2,2,2,2,2],n_neurons=512,conv_activation = "Mish", 
            dense_activation="SiLU", batch=True, dropout = 0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
epochs = 10
for epoch in range(epochs):  # change number of epochs as needed
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss = running_loss/len(train_loader)
    val_l,val_a = model.get_loss_accuracy(val_loader,criterion)

    print(f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_l:.4f} - "
            f"Val Accuracy: {val_a:.4f}")
    

# Initializing wandb
wandb.init(project="Assignment2_Q4")

test_l,test_a = model.get_loss_accuracy(test_loader,criterion)
wandb.log({"test_loss": test_l, "test_acc": test_a})


data_iter = iter(test_loader)
images, labels = next(data_iter)

# Taking the first 10 samples
images = images[:10]
labels = labels[:10]

# Getting Predictions for those 10 samples
model.eval()
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Move to CPU for visualization
images = images.cpu()
preds = list(preds.cpu().numpy())
labels = list(labels.cpu().numpy())


# Getting names of classes
d = {0:"Amphibia",1:"Animalia",2:"Arachnida",3:"Aves",4:"Fungi",5:"Insecta",6:"Mammalia",7:"Mollusca",8:"Plantae",9:"Reptilia"}
for i in range(len(labels)):
    labels[i] = d[labels[i]]
    preds[i] = d[preds[i]]


# Making a classification table
table = wandb.Table(columns=["Image", "True Label", "Predicted Label", "Result"])

for img, true_label, pred_label in zip(images, labels, preds):
    correct = (true_label == pred_label)
    status = "🟢 Correct" if correct else "🔴 Wrong"
    table.add_data(wandb.Image(img),true_label,pred_label,status)

# Logging the classification table
wandb.log({"Model Predictions": table})