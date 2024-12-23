import torch
import argparse
from dataset_utils import get_data_loaders
from model import SimpleCNN
from train import train_model
from test import test_model

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str,default="cuda", help='"cuda" or "cpu"')

    args = parser.parse_args()

    print("Args:", args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size
    )

    # Create model
    model = SimpleCNN(num_classes=10)
    model.to(device)

    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr
    )

    #Evaluate the model
    test_acc =test_model(trained_model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
    