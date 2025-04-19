import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,filters=list,kernel_sizes = list,pool_sizes = list,n_neurons=int,conv_activation = str, dense_activation = str,
                 batch=True,dropout=float):

        super().__init__()

        # Input Image Dimension
        self.img_dim = 224

        # Convolution Activations
        if conv_activation == "ReLU":
            self.conv_activation = nn.ReLU()
        if conv_activation == "GELU":
            self.conv_activation = nn.GELU()
        if conv_activation == "SiLU":
            self.conv_activation = nn.SiLU()
        if conv_activation == "Mish":
            self.conv_activation = nn.Mish()
        
        # Dense Activations
        if dense_activation == "ReLU":
            self.dense_activation = nn.ReLU()
        if dense_activation == "GELU":
            self.dense_activation = nn.GELU()
        if dense_activation == "SiLU":
            self.dense_activation = nn.SiLU()
        if dense_activation == "Mish":
            self.dense_activation = nn.Mish()
        
        # Convolution Block 1
        layers1 = [nn.Conv2d(3, filters[0], kernel_sizes[0])]
        if batch:
            layers1.append(nn.BatchNorm2d(filters[0]))
        layers1.append(self.conv_activation)
        layers1.append(nn.MaxPool2d(pool_sizes[0]))
        self.conv_block1 = nn.Sequential(*layers1)

        # Convolution Block 2
        layers2 = [nn.Conv2d(filters[0], filters[1], kernel_sizes[1])]
        if batch:
            layers2.append(nn.BatchNorm2d(filters[1]))
        layers2.append(self.conv_activation)
        layers2.append(nn.MaxPool2d(pool_sizes[1]))
        self.conv_block2 = nn.Sequential(*layers2)

        # Convolution Block 3
        layers3 = [nn.Conv2d(filters[1], filters[2], kernel_sizes[2])]
        if batch:
            layers3.append(nn.BatchNorm2d(filters[2]))
        layers3.append(self.conv_activation)
        layers3.append(nn.MaxPool2d(pool_sizes[2]))
        self.conv_block3 = nn.Sequential(*layers3)

        # Convolution Block 4
        layers4 = [nn.Conv2d(filters[2], filters[3], kernel_sizes[3])]
        if batch:
            layers4.append(nn.BatchNorm2d(filters[3]))
        layers4.append(self.conv_activation)
        layers4.append(nn.MaxPool2d(pool_sizes[3]))
        self.conv_block4 = nn.Sequential(*layers4)

        # Convolution Block 5
        layers5 = [nn.Conv2d(filters[3], filters[4], kernel_sizes[4])]
        if batch:
            layers5.append(nn.BatchNorm2d(filters[4]))
        layers5.append(self.conv_activation)
        layers5.append(nn.MaxPool2d(pool_sizes[4]))
        self.conv_block5 = nn.Sequential(*layers5)

        self._init_fc(3)

        # Dense Layer
        self.fc1 = nn.Linear(self.flattened_size,n_neurons)
        self.dropout = nn.Dropout(p=dropout)

        # Output Layer
        self.output = nn.Linear(n_neurons,10)

    # Function for getting dimension of x after convolution block 5
    def _init_fc(self, input_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.conv_block5(x)
            self.flattened_size = x.view(1, -1).shape[1]  # compute flattened size

    # Forward Pass
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = torch.flatten(x, 1)
        x = self.dense_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

    def get_loss_accuracy(self,dataloader,criterion):
        self.eval()
        val_loss = 0.0
        correct = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = self(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(dataloader.dataset)
        val_accuracy = correct / len(dataloader.dataset)
        return val_loss,val_accuracy
    
    def predict(self, dataloader):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Concatenating all batches into single tensors
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return all_preds, all_labels
