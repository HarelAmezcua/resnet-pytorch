import sys
import os
sys.path.append('..\\')
import torch
import multiprocessing as mp
from torchvision import transforms
import csv
from tdqm import tqdm

from src.utils import My_Resnet, My_Dataset

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        # Example: batch could be (images, labels)
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")  

    return loss.item()

def train_model(num_epochs=10):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create your dataset
    train_dataset = My_Dataset(r'C:\github\resnet-pytorch\dataset\train')
    val_dataset = My_Dataset(r'C:\github\resnet-pytorch\dataset\val')


    # Create DataLoader with multiple workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=2,  
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    # Create model, move to device
    model = My_Resnet().to(device)

    # Example optimizer, loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Create or check for csv file
    csv_file = 'training_loss.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])

    # Training loop
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # Save loss to csv
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, loss])
            
        # Optionally print or log training metrics
        print(f"Finished epoch {epoch + 1}")


def test_dataset_item():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    dataset = My_Dataset(r'C:\github\resnet-pytorch\dataset_2', transform)
    image, label = dataset[0]

    # Comparing the network output shape with the label
    model = My_Resnet()
    output = model(image.unsqueeze(0))
    print("Output: ",output.shape)    
    print("Original",image.shape, label.shape)

if __name__ == '__main__':
    mp.freeze_support()  # Needed for Windows multiprocessing
    train_model(num_epochs=50)