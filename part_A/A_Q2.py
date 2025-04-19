import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
import wandb
import yaml

from A_Q1 import CNN


# Function which runs during the wandb sweep
def train():
    wandb.init()

    # Getting all the configurations
    config = wandb.config

    n_filters = config.n_filters
    filter_size = config.filter_size
    filter_org = config.filter_org
    conv_activation_fun = config.conv_activation_func
    dense_activation_fun = config.dense_activation_func
    batch_normalization = config.batch_normalization
    dropout = config.dropout
    
    # Initializing wandb run
    wandb.run.name = f"n_filters{n_filters}_fsize_{filter_size}_conv_ac_{conv_activation_fun}_dense_ac_{dense_activation_fun}_batch_{batch_normalization}_dropout_{dropout}"
    wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filters = [n_filters]
    if filter_org == "same":
        for i in range(4):
            filters.append(n_filters)
    elif filter_org == "double":
        for i in range(4):
            filters.append(2*filters[i])
    if filter_org == "halve":
        for i in range(4):
            filters.append(filters[i]/2) 

    # Initializing the model
    kernel_sizes = [filter_size for _ in range(5)]
    model = CNN(filters=filters,kernel_sizes = kernel_sizes,pool_sizes = [2,2,2,2,2],n_neurons=128,conv_activation = conv_activation_fun, 
                dense_activation=dense_activation_fun, batch=batch_normalization, dropout = dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 10
    epo = []
    val_accuracy = []
    val_loss = []
    for epoch in range(epochs):  
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
        
        epo.append(epoch+1)
        val_loss.append(val_l)
        val_accuracy.append(val_a)

    # Logging all the metrics to wandb
    for i in range(len(epo)):
        wandb.log({"epochs": epo[i], "val_loss": val_loss[i], "val_accuracy": val_accuracy[i]})
        
    test_l,test_a = model.get_loss_accuracy(test_loader,criterion)
    wandb.log({"test_loss": test_l, "test_acc": test_a})

if __name__ == "__main__":

    # Defining the transform to imitate ImageNet dimensions.
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


    with open("part_A/config.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)

    # Defining the wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="Assignment2_Q3")
    wandb.agent(sweep_id,function=train,count=30)