import sys
import os
import torch
import numpy as np
import random
import csv
from tqdm import tqdm
import multiprocessing as mp
from torch.utils.data import DataLoader
from src.utils import My_Resnet, My_Dataset
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        if labels.ndim > 1:
            _, targets = torch.max(labels, 1)
        else:
            targets = labels

        correct += (preds == targets).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def train_model(num_epochs=50, patience=10):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = My_Dataset(r'C:\github\resnet-pytorch\dataset\train', type='train')
    val_dataset = My_Dataset(r'C:\github\resnet-pytorch\dataset\val', type='val')

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=2, shuffle=True,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2, shuffle=False,
                            pin_memory=True, persistent_workers=True)

    model = My_Resnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    csv_file = 'training_log.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping due to no improvement.")
                break


if __name__ == '__main__':
    mp.freeze_support()
    train_model(num_epochs=50)
