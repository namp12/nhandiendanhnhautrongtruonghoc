import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from src.core.dataset import VideoDataset, get_transform
from src.core.model import get_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def evaluate_model(model_path, data_dir, device):
    # Load dataset
    # Note: Using same transform as validation
    dataset = VideoDataset(data_dir, clip_len=16, transform=get_transform(mode='val'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(dataset)} clips from {data_dir}")
    print(f"Classes: {dataset.classes}")
    
    # Load model
    model = get_model(num_classes=len(dataset.classes))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Model file not found!")
        return

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.classes, yticklabels=dataset.classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.savefig(os.path.join(os.path.dirname(model_path), 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')}")

if __name__ == "__main__":
    MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
    # Use TEST set if available, otherwise VAL set
    TEST_DIR = r"e:\Violence_Detection_System\dataset_split\test"
    if not os.path.exists(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
        print("Test set not found or empty. Using Validation set.")
        TEST_DIR = r"e:\Violence_Detection_System\dataset_split\val"
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    evaluate_model(MODEL_PATH, TEST_DIR, DEVICE)
