import torch
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support as score
 
def train(params, DataClass, model, criterion, device = "cpu"):
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
    model.to(device)
    y_pred = torch.tensor([], device=device)
    y_true = torch.tensor([], device=device)

    with torch.no_grad():
        for _, metadata in enumerate(DataClass.dataloader['test']):
            inputs = metadata[0]
            labels = metadata[1]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            #Append to the mega-list
            y_pred = torch.cat((y_pred, predicted), dim = 0)
            y_true = torch.cat((y_true, labels), dim = 0)

    accuracy = (y_pred == y_true).sum().item() / y_true.size(0)
    print(f"Accuracy on test set: {accuracy:.4f}")
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    
    sc = score(y_true, y_pred)
    #verbose prints
    print(f"Accuracy on test set: {accuracy:.4f}")
    print('Precision: {}'.format(sc[0]))
    print('Recall: {}'.format(sc[1]))
    print('F1-Score: {}'.format(sc[2]))