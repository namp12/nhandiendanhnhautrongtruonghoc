import torch
import cv2
import numpy as np
from .model import get_model
import os

class ViolenceDetector:
    def __init__(self, model_path, device=None, num_frames=16):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        
        # Load Model
        self.classes = ['chay', 'danh_nhau', 'di', 'dung', 'nam', 'nga', 'ngoi', 'normal'] # Updated with 'normal' class
        # Note: Order must match dataset.classes. Assuming alphabetical for now.
        # Ideally save/load classes from a file.
        
        self.model = get_model(num_classes=len(self.classes), pretrained=False)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
             print(f"Warning: Model path {model_path} not found. Using random weights.")
             
        self.model.to(self.device)
        self.model.eval()
        
        self.transform_mean = np.array([0.43216, 0.394666, 0.37645])
        self.transform_std = np.array([0.22803, 0.22145, 0.216989])

    def preprocess_clip(self, frames):
        """
        Args:
            frames: List of frames (H, W, C) in RGB
        Returns:
            tensor: (1, C, T, H, W) normalized
        """
        # Resize to 128x128 first (or whatever model trained on)
        frames_resized = [cv2.resize(f, (128, 128)) for f in frames]
        
        # To Tensor (T, H, W, C) -> (C, T, H, W)
        buffer = np.array(frames_resized).astype(np.float32) / 255.0
        
        # Normalize?
        # buffer = (buffer - self.transform_mean) / self.transform_std
        
        buffer = torch.from_numpy(buffer).permute(3, 0, 1, 2)
        return buffer.unsqueeze(0).to(self.device)

    def predict(self, clip):
        """
        Predicts action for a single clip (list of frames)
        """
        with torch.no_grad():
            inputs = self.preprocess_clip(clip)
            outputs = self.model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            label = self.classes[pred_idx.item()]
            confidence = conf.item()
            return label, confidence
