import os
import shutil
import random
import math

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Get all classes
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found classes: {classes}")

    for cls in classes:
        print(f"Processing class: {cls}")
        cls_source_dir = os.path.join(source_dir, cls)
        
        # Create class directories in output
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        files = [f for f in os.listdir(cls_source_dir) if f.lower().endswith(('.avi', '.mp4', '.mov'))]
        random.shuffle(files)

        total_files = len(files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count

        train_files = files[:train_count]
        val_files = files[train_count:train_count+val_count]
        test_files = files[train_count+val_count:]

        print(f"  - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        # Copy files
        for f in train_files:
            shutil.copy2(os.path.join(cls_source_dir, f), os.path.join(train_dir, cls, f))
        
        for f in val_files:
            shutil.copy2(os.path.join(cls_source_dir, f), os.path.join(val_dir, cls, f))

        for f in test_files:
            shutil.copy2(os.path.join(cls_source_dir, f), os.path.join(test_dir, cls, f))

def main():
    source_dataset = r"e:\Violence_Detection_System\dataset\train"
    output_dataset = r"e:\Violence_Detection_System\dataset_split"
    
    # Check if source exists
    if not os.path.exists(source_dataset):
        print(f"Source directory not found: {source_dataset}")
        return

    print("Starting dataset split...")
    split_dataset(source_dataset, output_dataset)
    print("\nDataset split completed successfully!")
    print(f"Output directory: {output_dataset}")

if __name__ == "__main__":
    main()
