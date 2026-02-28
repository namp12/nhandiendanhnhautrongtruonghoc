import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model import get_model
from src.core.dataset import VideoDataset, get_transform

def evaluate_model(model_path, data_dir, train_dir, device):
    # Get classes from train directory
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(train_classes)
    
    # Dataset and Loader (using train_classes to maintain index mapping)
    transform = get_transform(mode='val')
    test_ds = VideoDataset(data_dir, clip_len=16, transform=transform)
    
    # Manually override class mapping to match train set
    test_ds.classes = train_classes
    test_ds.class_to_idx = {cls_name: i for i, cls_name in enumerate(train_classes)}
    # Re-build samples with correct label indices
    test_ds.samples = []
    for cls_name in os.listdir(data_dir):
        if cls_name in test_ds.class_to_idx:
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for file_name in os.listdir(cls_dir):
                    if file_name.lower().endswith(('.avi', '.mp4', '.mov')):
                        test_ds.samples.append((os.path.join(cls_dir, file_name), test_ds.class_to_idx[cls_name]))
    
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2) # Reduced batch size to be safer
    
    # Load Model
    model = get_model(num_classes=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Evaluating {len(test_ds)} videos...")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(test_loader)} done.")

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("="*30)
    print("\nConfusion Matrix (Normalized):")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(np.round(cm_norm, 2))
    print("\nLabels Order:", train_classes)


if __name__ == "__main__":
    MODEL_PATH = r"e:\Violence_Detection_System\models\best_model.pth"
    DATA_DIR = r"e:\Violence_Detection_System\dataset_split\test"
    TRAIN_DIR = r"e:\Violence_Detection_System\dataset_split\train"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    evaluate_model(MODEL_PATH, DATA_DIR, TRAIN_DIR, device)

