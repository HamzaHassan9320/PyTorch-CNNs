import torch
import torch.nn as nn

def test_model(model, test_loader,device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct=0
    total=0
    test_loss=0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels =images.to(device), labels.to(device)
            outputs =model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted =torch.max(outputs,1)
            correct+= (predicted==labels).sum().item()
            total+=labels.size(0)

    avg_loss = test_loss/total
    acc= 100.0 * correct/total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%")
    return acc