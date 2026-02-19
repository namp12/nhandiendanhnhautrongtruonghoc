import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def get_model(num_classes=2, pretrained=True):
    """
    Returns a R3D-18 model for video classification.
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load weights pretrained on Kinetics-400.
    """
    if pretrained:
        weights = R3D_18_Weights.DEFAULT
    else:
        weights = None
        
    model = r3d_18(weights=weights)
    
    # Modify the last fully connected layer to match our number of classes
    # Original layer: (fc): Linear(in_features=512, out_features=400, bias=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Test the model
    net = get_model(num_classes=2)
    print(net)
    
    # Test with a dummy input (Batch, Channels, Frames, Height, Width)
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    output = net(dummy_input)
    print(f"Output shape: {output.shape}")
