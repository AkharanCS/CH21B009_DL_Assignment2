import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,random_split
import wandb
import yaml

# Defining the transform to match the input size of ImageNet
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
test_loader = DataLoader(dataset_test)

# Loading the pretrained ResNet50
model = models.resnet50(pretrained=True)

# Freezing early layers
for param in model.parameters():
    param.requires_grad = False

# Replacing the final layer with a layer of size 10
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Unfreezing the last layer
for param in model.fc.parameters():
    param.requires_grad = True


# Setting up loss and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

epochs = 10
# Finetuning the model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)

    train_loss = running_loss/len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss/len(val_loader):.4f} - "
              f"Val Acc: {val_acc:.4f}")