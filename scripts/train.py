import sys
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path so we can import src.core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model import get_model
from src.core.dataset import VideoDataset

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, save_path='models/best_model.pth'):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    # Configuration
    DATA_DIR = r"e:\Violence_Detection_System\dataset_split"
    MODEL_SAVE_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
    BATCH_SIZE = 8  # Adjust based on GPU memory (RTX 3050 6GB can handle 8-16 probably)
    NUM_EPOCHS = 10 # Start with 10 for quick test, increase later
    LEARNING_RATE = 0.001
    NUM_FRAMES = 16 # C3D/R3D usually takes 16 frames
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    image_datasets = {x: VideoDataset(os.path.join(DATA_DIR, x), clip_len=NUM_FRAMES) for x in ['train', 'val']}
    
    # Dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    num_classes = len(image_datasets['train'].classes)
    print(f"Classes: {image_datasets['train'].classes}")

    # Model
    print("Loading model...")
    model_ft = get_model(num_classes=num_classes, pretrained=True)
    model_ft = model_ft.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

    # Train
    print("Starting training...")
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
        
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=NUM_EPOCHS, save_path=MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
