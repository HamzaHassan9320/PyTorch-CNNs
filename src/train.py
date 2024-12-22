import torch
import time

import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, device, epochs=10, lr=0.01, momentum=0.9, weight_decay=5e-4, step_size=20, gamma=0.1):
    """
    Train the model on train_loader for the specified number of epochs.
    Returns the trained model.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr =lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        correct=0
        total=0

        start_time=time.time()

        for batch_idx, (images, labels)  in enumerate(train_loader):
            images,labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss +=loss.item() * images.size(0)
            _, predicted =torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total +=labels.size(0)

        scheduler.step()

        epoch_loss = running_loss/total
        epoch_acc = 100.0 * correct / total
        end_time = time.time()

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {epoch_loss:.4f}, "
              f"Acc: {epoch_acc:.2}%, "
              f"Time: {(end_time - start_time):.2f}s")
        
    return model