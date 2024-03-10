import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data_gen import make_dataset
 
def train(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    num_classes = 7
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return model

def test(test_dataset, model): 
    test_loader = DataLoader(test_dataset, batch_size=64) 
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.4f}")

if __name__ == "__main__":
    image_folder = 'aiml-general-championship/KCDH2024_Training_Input_10K/KCDH2024_Training_Input_10K'
    gt_file = 'aiml-general-championship/KCDH2024_Training_GroundTruth.csv'
    mapping_file = 'disease_id.json'

    dataset = make_dataset(image_folder, gt_file, mapping_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(dataset["train"])
    test(dataset["test"], model)
