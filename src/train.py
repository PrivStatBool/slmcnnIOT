import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_grocery_data
from model import get_resnet34
import matplotlib.pyplot as plt
import os

def train_model(num_epochs=10, learning_rate=0.001, batch_size=32, data_dir='../data/', save_path='../outputs/models/resnet34_grocery.pth'):
    # Load dataset
    train_loader, val_loader, _ = load_grocery_data(batch_size=batch_size, data_dir=data_dir, augment=True)

    # Initialize the model, loss function, and optimizer
    model = get_resnet34(num_classes=81)
    
    # No freezing here - train all layers
    criterion = nn.CrossEntropyLoss()

    # The optimizer updates all parameters (no layers are frozen)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss and accuracy for each epoch
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagate for all layers
            optimizer.step()  # Update the weights
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Plot loss and accuracy
    plot_results(train_losses, val_losses, val_accuracies)

def plot_results(train_losses, val_losses, val_accuracies):
    # Use an absolute path for saving the figure
    output_dir = '../outputs/figs'
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)  # Create a list of epoch numbers starting from 1

    # Plot loss
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xticks(epochs)  # Set the x-axis ticks to the epoch numbers
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xticks(epochs)  # Set the x-axis ticks to the epoch numbers
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    print('Training results saved as outputs/figs/training_results.png')


if __name__ == '__main__':
    train_model()

