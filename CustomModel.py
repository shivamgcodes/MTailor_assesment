import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomModel(nn.Module):
    def __init__(self, mtailor):
        super(CustomModel, self).__init__()
        self.mtailor = mtailor
    
    def preprocess_torch(self, x):
        x = F.interpolate(x, size=(224, 224), align_corners=False, mode = "bilinear", antialias=False)
        #tensor is already in range[0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)

        x = (x - mean) / std
        return x

    def forward(self, x):
        x = self.preprocess_torch(x)
        return self.mtailor.forward(x)
    
