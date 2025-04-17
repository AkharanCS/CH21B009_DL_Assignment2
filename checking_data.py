from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split


transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize all images to 224x224
    transforms.ToTensor(),                    # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = "inaturalist_12K/train"

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split into training and validation sets


val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(len(train_dataset))