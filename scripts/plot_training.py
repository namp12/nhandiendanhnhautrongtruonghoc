import matplotlib.pyplot as plt
import re
import sys
import os

def plot_logs(log_file):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    epochs = []

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Parse content
    # Look for lines like: "train Loss: 0.1234 Acc: 0.9876"
    # and "val Loss: 0.2345 Acc: 0.8765"
    
    # We assume they appear in order: train then val for each epoch
    
    train_matches = re.findall(r'train Loss: ([\d\.]+) Acc: ([\d\.]+)', content)
    val_matches = re.findall(r'val Loss: ([\d\.]+) Acc: ([\d\.]+)', content)
    
    for i, (loss, acc) in enumerate(train_matches):
        train_loss.append(float(loss))
        train_acc.append(float(acc))
        epochs.append(i)
        
    for i, (loss, acc) in enumerate(val_matches):
        # Handle case where val might be missing for last epoch if interrupted
        if i < len(epochs):
            val_loss.append(float(loss))
            val_acc.append(float(acc))

    # Plot
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Acc')
    if val_acc:
        plt.plot(epochs[:len(val_acc)], val_acc, label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    if val_loss:
        plt.plot(epochs[:len(val_loss)], val_loss, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    output_file = log_file + '.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Default log file
    LOG_FILE = r"e:\Violence_Detection_System\train_augmented.log"
    
    # Or existing train.log
    if not os.path.exists(LOG_FILE) and os.path.exists(r"e:\Violence_Detection_System\train.log"):
        LOG_FILE = r"e:\Violence_Detection_System\train.log"
        
    # Check if user passed arguments
    if len(sys.argv) > 1:
        LOG_FILE = sys.argv[1]
        
    print(f"Parsing {LOG_FILE}...")
    plot_logs(LOG_FILE)
