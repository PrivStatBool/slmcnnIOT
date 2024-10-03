import os

def get_classes(directory_path):
    """Return the list of classes (subdirectories) in the given directory."""
    if os.path.exists(directory_path):
        return set([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
    return set()

def find_missing_classes(data_dir='data/'):
    """Find missing classes in train, val, or test sets."""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Get the set of classes in each split
    train_classes = get_classes(train_dir)
    val_classes = get_classes(val_dir)
    test_classes = get_classes(test_dir)
    
    all_classes = train_classes | val_classes | test_classes
    
    # Find missing classes
    missing_in_train = all_classes - train_classes
    missing_in_val = all_classes - val_classes
    missing_in_test = all_classes - test_classes
    
    print(f"Classes missing in train: {missing_in_train}")
    print(f"Classes missing in val: {missing_in_val}")
    print(f"Classes missing in test: {missing_in_test}")

if __name__ == "__main__":
    find_missing_classes(data_dir='../data/')

