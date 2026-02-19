import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None, return_path=False):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. 'dataset/train').
            clip_len (int): Number of frames to sample per video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.return_path = return_path
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(('.avi', '.mp4', '.mov')):
                    self.samples.append((os.path.join(cls_dir, file_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        frames = self._load_video(video_path)
        
        # Format for PyTorch (C, T, H, W) where T is temporal dimension (frames)
        # Frames are currently (T, H, W, C)
        
        # Apply transforms if any
        if self.transform:
            frames = self.transform(frames)
        else:
             # Convert to tensor and permute to (C, T, H, W)
             # Default transform: Normalize to [0, 1]
             frames = torch.from_numpy(frames).float() / 255.0
             frames = frames.permute(3, 0, 1, 2) # (C, T, H, W)

        if self.return_path:
            return frames, label, video_path
        
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # If video is too short, loop it
            # If too long, sample uniformly or take first N
            
            # Simple strategy: Read all frames, then sample
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to expected input size (e.g. 112x112) or training size (128x171)
                # We do transform outside usually, but let's resize here to be safe and save memory
                frame = cv2.resize(frame, (128, 128)) 
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
             # Return black frames if failed
             return np.zeros((self.clip_len, 128, 128, 3), dtype=np.uint8)

        # Sampling logic
        # If we need 'clip_len' frames
        buffer = np.array(frames)
        
        if len(buffer) < self.clip_len:
            # Loop the video to fill
            indices = np.resize(np.arange(len(buffer)), self.clip_len)
            buffer = buffer[indices]
        elif len(buffer) > self.clip_len:
            # Uniform sampling
            indices = np.linspace(0, len(buffer) - 1, self.clip_len).astype(int)
            buffer = buffer[indices]
            
        return buffer

    

def get_transform(mode='train'):
    import torchvision.transforms as T
    # Simple transform: ToTensor + Normalize
    # In a real scenario, we might want RandomCrop, Flip for train
    # But for Video, we need to apply to all frames consistently.
    # Here we just return a lambda/function that processes the clip (T, H, W, C)
    
    def transform_fn(clip):
        # clip is np.array (T, H, W, C) [0-255]
        # Convert to Tensor (C, T, H, W) [0-1]
        buffer = torch.from_numpy(clip).float() / 255.0
        buffer = buffer.permute(3, 0, 1, 2) # (C, T, H, W)
        
        # Normalize (approximate statistics for Kinetics-400)
        # mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        # std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        # buffer = (buffer - mean) / std
        return buffer
        
    return transform_fn

if __name__ == "__main__":
    # Test dataset
    ds = VideoDataset(r"e:\Violence_Detection_System\dataset_split\train", transform=get_transform())
    if len(ds) > 0:
        data, label = ds[0]
        print(f"Data shape: {data.shape}, Label: {label}")
