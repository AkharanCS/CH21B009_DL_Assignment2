import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
from matplotlib import pyplot as plt

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
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
data_iter = iter(val_loader)
images, labels = next(data_iter)

# Take the first 10 samples
images = images[:10]
labels = labels[:10]

preds = [0,2,2,1,2,3,4,5,6,7]
# Move to CPU for visualization

# Sample test images and predictions (replace with actual data)
# Assuming: images shape (N, H, W, C), y_true and y_pred are label arrays


# Create 10x3 grid
fig, axes = plt.subplots(10, 3, figsize=(12, 20))
fig.suptitle('Model Predictions vs Ground Truth', fontsize=16)
for i in range(10):
    ax = axes.flat[i]
    img = images[i]
    true_label = labels[i]
    pred_label = preds[i]
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    img = img.permute(1, 2, 0) * std + mean  # denormalize
    img = img.clamp(0, 1).numpy()
    ax.imshow(img)
    
    ax.axis('off')
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}",
                fontsize=8, color='green' if pred_label == true_label else 'red')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
plt.show()


