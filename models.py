import torch
import torchvision.models as models
 
def train(params, device, DataClass):
    num_classes = 7
    model = models.resnet18(params['pretrained'])

    #Adding the FC layer on top
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    torch.nn.init.xavier_uniform_(model.fc.weight)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    model.to(device)

    for epoch in range(params['num_epochs']):
        model.train()
        running_loss = 0.0
        for _, metadata in enumerate(DataClass.dataloader['train']):
            inputs = metadata[0]
            labels = metadata[1]
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        scheduler.step()
        epoch_loss = running_loss / len(DataClass.dataset['train'])
        print(f"Epoch {epoch+1}/{params['num_epochs']}, Loss: {epoch_loss:.4f}")
    
    return model

def test(DataClass, model, device): 
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _, metadata in enumerate(DataClass.dataloader['test']):
            inputs = metadata[0]
            labels = metadata[1]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.4f}")