import sys, os, torch
import numpy as np
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.model import get_model
from src.core.dataset import VideoDataset, get_transform
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"e:\Violence_Detection_System\models\best_model.pth"
    data_dir = r"e:\Violence_Detection_System\dataset_split\test"
    train_dir = r"e:\Violence_Detection_System\dataset_split\train"
    
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(train_classes)
    
    test_ds = VideoDataset(data_dir, clip_len=16, transform=get_transform(mode='val'))
    test_ds.classes = train_classes
    test_ds.class_to_idx = {cls_name: i for i, cls_name in enumerate(train_classes)}
    
    test_ds.samples = []
    for cls_name in os.listdir(data_dir):
        if cls_name in test_ds.class_to_idx:
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for file_name in os.listdir(cls_dir):
                    if file_name.lower().endswith(('.avi', '.mp4', '.mov')):
                        test_ds.samples.append((os.path.join(cls_dir, file_name), test_ds.class_to_idx[cls_name]))
    
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    print(f"Total videos: {len(test_ds)}")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            outputs = model(inputs.to(device))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
            if (i+1) % 20 == 0: print(f"Batch {i+1} done.")
            
    print("\n" + "="*50)
    print(classification_report(all_labels, all_preds, labels=range(len(train_classes)), target_names=train_classes, digits=4, zero_division=0))
    print("="*50)

if __name__ == '__main__': main()
