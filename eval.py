import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
from torchvision.models import resnet18

def get_test_dataset(test_path):
    transform = transforms.Compose([
                        transforms.Resize((224, 224)), 
                        #transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.2),
                        #transforms.RandomRotation(degrees=90),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])
    
    test_dataset = ImageFolder(test_path, transform=transform)
    image_ids = [file[0] for file in test_dataset.imgs]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return test_loader, image_ids

def generate_predictions(model_path, test_loader): 
    num_classes = 7
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predictions = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
    
    return predictions

if __name__=='main':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = 'aiml-general-championship/KCDH2024_Test_Input/KCDH2024_Test_Input'
    test_loader, image_ids = get_test_dataset(test_path)
    model_path = 'ResNet18.pth'
    predictions = generate_predictions(model_path, test_loader)
    submission_df = pd.DataFrame({'Image_ID': image_ids, 'Prediction': predictions})
    submission_df.to_csv('submission.csv', index=False)