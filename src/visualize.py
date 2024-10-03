import os
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset import load_grocery_data

# Function to save a grid of images with specific product names underneath
def imshow_with_product_names(images, image_paths, filename='output.png'):
    # Set up the figure and axis for subplots
    num_images = len(images)
    cols = 8  # You can adjust this to change the number of columns
    rows = (num_images + cols - 1) // cols  # Calculate rows based on number of images and columns

    fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))  # Dynamic figure size based on rows
    axes = axes.flatten()  # Flatten the 2D axes array for easy iteration

    # Loop through each image, plot it, and add the specific product name underneath
    for i, (img, ax) in enumerate(zip(images, axes)):
        img = img / 2 + 0.5  # Unnormalize the image
        npimg = img.numpy()  # Convert from tensor to NumPy array
        ax.imshow(np.transpose(npimg, (1, 2, 0)))  # Rearrange dimensions (H, W, C)

        ax.axis('off')  # Hide axes

    # If there are more axes than images, hide the remaining empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Save the image grid to a file
    plt.tight_layout()
    output_dir = os.path.expanduser('~/dev/cnn/cnn_project/outputs/figs')
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Image saved as {os.path.join(output_dir, filename)}")
    plt.close()

# Visualize a batch of classifier items and save the figure with product names
def visualize_classifier_items_with_product_names(data_loader, num_images=8, output_filename="classifier_output.png"):
    # Get a batch of data from the loader
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Select the first 'num_images' images
    images = images[:num_images]
    image_paths = [data_loader.dataset.imgs[i][0] for i in range(num_images)]

    # Show or save the images with product names
    imshow_with_product_names(images, image_paths, filename=output_filename)

if __name__ == '__main__':
    # Define the transformations used to prepare your dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the dataset (e.g., your grocery dataset)
    data_dir = '../data/test'  # Path to your test dataset
    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Create a DataLoader to load the test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # Visualize classifier items and save the output image with product names
    visualize_classifier_items_with_product_names(test_loader, num_images=32, output_filename="classifier_output.png")

