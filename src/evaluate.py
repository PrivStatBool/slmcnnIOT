import torch
from dataset import load_grocery_data
from model import get_resnet34

def evaluate_model(model_path='outputs/models/resnet34_grocery.pth', batch_size=32, data_dir='../data/'):
    # Load dataset
    _, _, test_loader = load_grocery_data(batch_size=batch_size, data_dir=data_dir)

    # Print class names to ensure they are loaded correctly
    class_names = test_loader.dataset.classes
    print(f"Class Names: {class_names}")

if __name__ == '__main__':
    evaluate_model()

