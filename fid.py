import os
from PIL import Image
import numpy as np
import torch
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Define data paths
real_images_dir = "E:\\Msc\\Lab\\data\\fid_data\\original_data"

# Define transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
dataset = datasets.ImageFolder(root=real_images_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Map class indices to names (useful for FID computation later)
class_names = dataset.classes
print("Class names:", class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load a pre-trained ResNet18 model
'''
model = resnet18(pretrained=True)
model.fc = nn.Identity()  # Remove final classification layer for feature extraction
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Not needed during feature extraction but used for training
epochs = 10

# Train the feature extractor
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, _ in train_loader:  # `_` because we're not using labels for feature extraction
        inputs = inputs.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Fake labels for loss computation
        labels = torch.zeros(outputs.size(0), dtype=torch.long).to(device)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained feature extractor
torch.save(model.state_dict(), "E:\\Msc\\Lab\\spectrum_sharing_with_stable_diffusion\\diffusion_model\\cnn_weights\\custom_feature_extractor.pth")
print("Feature extractor trained and saved.")'''

###################################################################################################################################

def calculate_fid(real_dir, generated_dir, model,device):
    """
    Calculate FID score between real and generated images using the custom model.
    """
    # Extract features for real images
    real_features = extract_features_from_directory(real_dir, model, device)
    real_mu, real_sigma = calculate_statistics(real_features)

    # Extract features for generated images
    gen_features = extract_features_from_directory(generated_dir, model, device)
    gen_mu, gen_sigma = calculate_statistics(gen_features)

    # Calculate FID
    fid = calculate_fid_with_regularization(real_mu, real_sigma, gen_mu, gen_sigma)
    return fid


def extract_features_from_directory(image_dir, model, device):
    """
    Extract features for all images in a directory using the custom model.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model.eval()
    features = []

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(image).squeeze().cpu().numpy()
                features.append(feature)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    return np.array(features)

def calculate_statistics(features):
    """
    Calculate mean and covariance of features.
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid_with_regularization(mu_real, sigma_real, mu_gen, sigma_gen, eps=1e-6):
    if len(sigma_real.shape) == 2 and sigma_real.shape[0] == sigma_real.shape[1]:
        # Regularize the covariance matrix if it is 2D
        sigma_real += np.eye(sigma_real.shape[0]) * eps
    else:
        print(f"Shape of sigma_real: {sigma_real.shape}")
        print("Error: sigma_real is not a 2D square matrix!")
        return None

    sigma_gen += np.eye(sigma_gen.shape[0]) * eps
    # Compute the difference in means
    mean_diff = mu_real - mu_gen

    # Compute the square root of the product of covariance matrices
    cov_sqrt, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Compute FID
    fid = np.sum(mean_diff**2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fid

def compute_classwise_fid(real_base_dir, generated_base_dirs, class_names, model):
    """
    Compute FID score for each class for each model.
    """
    fid_scores = {model_name: [] for model_name in generated_base_dirs.keys()}
    for cls in class_names:
        real_class_dir = os.path.join(real_base_dir, cls)
        for model_name, gen_base_dir in generated_base_dirs.items():
            gen_class_dir = os.path.join(gen_base_dir, cls)
            fid = calculate_fid(real_class_dir, gen_class_dir, model,device)
            if fid is not None:
                fid_scores[model_name].append(fid)
            else:
                print(f"Skipping class {cls} for model {model_name} due to error.")
    return fid_scores

def plot_fid_scores(fid_scores, class_names):
    """
    Plot FID scores for each class across models.
    """
    x = np.arange(len(class_names))
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, scores) in enumerate(fid_scores.items()):
        ax.bar(x + i * width, scores, width, label=model_name)

    ax.set_xlabel('Classes')
    ax.set_ylabel('FID Score')
    ax.set_title('FID Score Comparison Across Models')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Class names (folders)
    class_names = [
        'ch1_empty_ch2_empty','ch1_collision_ch2_empty', 'ch1_collision_ch2_secondary', 'ch1_empty_ch2_collision',
        'ch1_empty_ch2_primary', 'ch1_empty_ch2_secondary', 'ch1_primary_ch2_collision',
        'ch1_primary_ch2_empty', 'ch1_primary_ch2_primary', 'ch1_primary_ch2_secondary',
        'ch1_secondary_ch2_empty', 'ch1_secondary_ch2_primary'
    ]
    
    fake_class_names= [
    'Class 0', 'Class 1', 'Class 2',
    'Class 3', 'Class 4', 'Class 5',
    'Class 6', 'Class 7', 'Class 8',
    'Class 9', 'Class 10', 'Class 11'
    ]

    real_images_dir = "E:\\Msc\\Lab\\data\\fid_data\\original_data"
    generated_images_dirs = {
        "GAN": "E:\\Msc\\Lab\\data\\fid_data\\GAN",
        "DDPM": "E:\\Msc\\Lab\\data\\fid_data\\DDPM",
        
    }

    # Load the custom feature extractor
    custom_model = resnet18(pretrained=False)
    custom_model.fc = nn.Identity()
    custom_model.load_state_dict(torch.load("E:\\Msc\\Lab\\spectrum_sharing_with_stable_diffusion\\diffusion_model\\cnn_weights\\custom_feature_extractor.pth"))
    custom_model = custom_model.to(device)

    # Compute FID scores
    fid_scores = compute_classwise_fid(real_images_dir, generated_images_dirs, class_names, custom_model)

    # Print average FID scores
    for model_name, scores in fid_scores.items():
        avg_fid = np.nanmean(scores)
        print(f"Average FID score for {model_name}: {avg_fid}")

    # Plot FID scores
    plot_fid_scores(fid_scores, fake_class_names)


if __name__ == "__main__":
    main()