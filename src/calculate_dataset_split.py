import os

def count_images_in_directory(directory_path):
    """Count the total number of images in a directory (including subdirectories)."""
    total_images = 0
    for root, _, files in os.walk(directory_path):
        total_images += len([f for f in files if f.endswith('.jpg') or f.endswith('.png')])
    return total_images

def count_classes_in_directory(directory_path):
    """Count the number of classes (subdirectories) in a directory."""
    if os.path.exists(directory_path):
        return len([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
    return 0

def calculate_dataset_split(data_dir='data/'):
    """Calculate the proportion of train, val, and test images, and count the number of classes."""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Count the images
    train_images = count_images_in_directory(train_dir)
    val_images = count_images_in_directory(val_dir)
    test_images = count_images_in_directory(test_dir)
    
    # Count the classes
    train_classes = count_classes_in_directory(train_dir)
    val_classes = count_classes_in_directory(val_dir)
    test_classes = count_classes_in_directory(test_dir)
    
    total_images = train_images + val_images + test_images
    
    # Calculate proportions
    train_ratio = (train_images / total_images) * 100 if total_images > 0 else 0
    val_ratio = (val_images / total_images) * 100 if total_images > 0 else 0
    test_ratio = (test_images / total_images) * 100 if total_images > 0 else 0
    
    print(f"Train set: {train_images} images ({train_ratio:.2f}%), {train_classes} classes")
    print(f"Validation set: {val_images} images ({val_ratio:.2f}%), {val_classes} classes")
    print(f"Test set: {test_images} images ({test_ratio:.2f}%), {test_classes} classes")
    print(f"Total images: {total_images}")

if __name__ == "__main__":
    calculate_dataset_split(data_dir='../data/')

