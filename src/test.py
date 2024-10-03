import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from model import get_resnet34
import os
import random

# Function to extract the fine-grained product name from the image path
def get_fine_grained_label(image_path):
    parts = image_path.split(os.sep)
    return parts[-2]  # Extract the fine-grained class label

# Function to predict product names for a batch of images
def predict_product_names(model, data_loader, num_images=5):
    # Collect image paths and labels
    indices = list(range(len(data_loader.dataset.samples)))
    random.shuffle(indices)
    selected_indices = indices[:num_images]
    
    # Display the image paths
#    for idx in selected_indices:
#        image_path = data_loader.dataset.samples[idx][0]
#        print(f"Processing image: {image_path}")

    # Prepare images
    images = torch.stack([data_loader.dataset[idx][0] for idx in selected_indices])

    # Pass the images through the model to get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the index of the highest prediction

    # Map predicted indices back to image paths for fine-grained labels
    product_names = []
    for idx in selected_indices:
        image_path = data_loader.dataset.samples[idx][0]
        product_name = get_fine_grained_label(image_path)
        product_names.append(product_name)

    return product_names

# Main function for testing
def main(num_images=5, model_path='../outputs/models/resnet34_grocery.pth', data_dir='../data/test'):
    # Initialize the model and load the trained weights
    model = get_resnet34(num_classes=81)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Define the transformations for the test set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Use SubsetRandomSampler to shuffle across the entire dataset
    sampler = SubsetRandomSampler(range(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, batch_size=len(test_dataset))

    # Predict the product names for 'num_images' images
    predicted_product_names = predict_product_names(model, test_loader, num_images=num_images)

    # Output the predicted product names
    print(predicted_product_names)

if __name__ == '__main__':
    main()

