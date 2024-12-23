import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# Define CBAM module
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss

# Paths to datasets
original_data_path = "E:\\Msc\\Lab\\data\\fid_data\\original_data"
generated_data_paths = {
    "GAN": "E:\\Msc\\Lab\\data\\fid_data\\GAN",
    "DDPM": "E:\\Msc\\Lab\\data\\fid_data\\DDPM",
    "VQ-VAE": "E:\\Msc\\Lab\\data\\fid_data\\VQ_VAE"
}

# Data transformations
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define function to combine datasets
def combine_datasets(original_path, generated_path, combined_path):
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)

    for cls in os.listdir(original_path):
        orig_class_dir = os.path.join(original_path, cls)
        gen_class_dir = os.path.join(generated_path, cls)
        combined_class_dir = os.path.join(combined_path, cls)

        if not os.path.exists(combined_class_dir):
            os.makedirs(combined_class_dir)

        for file in os.listdir(orig_class_dir):
            os.link(os.path.join(orig_class_dir, file), os.path.join(combined_class_dir, file))

        if os.path.exists(gen_class_dir):
            for file in os.listdir(gen_class_dir):
                os.link(os.path.join(gen_class_dir, file), os.path.join(combined_class_dir, file))

# Combine datasets
combined_data_paths = {name: f"E:\\Msc\\Lab\\data\\fid_data\\combined_{name}" for name in generated_data_paths.keys()}
for name, path in generated_data_paths.items():
    combine_datasets(original_data_path, path, combined_data_paths[name])

# Split dataset into train and test
def split_dataset(data_path, train_path, test_path, test_size=0.2):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for cls in os.listdir(data_path):
        class_dir = os.path.join(data_path, cls)
        train_class_dir = os.path.join(train_path, cls)
        test_class_dir = os.path.join(test_path, cls)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        files = os.listdir(class_dir)
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

        for file in train_files:
            os.link(os.path.join(class_dir, file), os.path.join(train_class_dir, file))

        for file in test_files:
            os.link(os.path.join(class_dir, file), os.path.join(test_class_dir, file))

split_data_paths = {}
for name, combined_path in combined_data_paths.items():
    train_path = f"E:\\Msc\\Lab\\data\\fid_data\\train_{name}"
    test_path = f"E:\\Msc\\Lab\\data\\fid_data\\test_{name}"
    split_dataset(combined_path, train_path, test_path)
    split_data_paths[name] = (train_path, test_path)

# Class names
train_dataset = datasets.ImageFolder(root=original_data_path, transform=transform)
class_names = train_dataset.classes

# Define function to train and evaluate
def train_and_evaluate(train_path, test_path, class_names, transform, device, epochs=10):
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.layer4.add_module('cbam', CBAM(channel=2048))
    model = model.to(device)

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate class-wise accuracy
    def classwise_accuracy(model, dataloader):
        model.eval()
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)

                for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    class_total[class_names[label]] += 1
                    if label == pred:
                        class_correct[class_names[label]] += 1

        return {cls: (class_correct[cls] / class_total[cls]) * 100 if class_total[cls] > 0 else 0 for cls in class_names}

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return classwise_accuracy(model, test_loader)

# Train and evaluate on combined datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_classwise_accuracies = {}
for name, (train_path, test_path) in split_data_paths.items():
    print(f"Training and evaluating for: Original + {name}")
    combined_classwise_accuracies[name] = train_and_evaluate(train_path, test_path, class_names, transform, device)

# Plot accuracies
def plot_accuracies(combined_classwise_accuracies, class_names):
    color_map = {
    "GAN": "#d95f02",       # Orange
    "DDPM": "#7570b3",      # Purple
    "VQ-VAE": "#e7298a"     # Pink
}
    x = np.arange(len(class_names))
    for name, acc in combined_classwise_accuracies.items():
        plt.plot(x, [acc[cls] for cls in class_names], marker='o', label=f"Original + {name}",color=color_map[name], linewidth=2)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy (%)")
    plt.title("Class-wise Accuracy Comparison")
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_accuracies(combined_classwise_accuracies, class_names) 