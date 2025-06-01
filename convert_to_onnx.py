from pytorch_model import Classifier, BasicBlock
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
    

model = Classifier(BasicBlock, [2,2,2,2])
model.load_state_dict(torch.load("model_weights/pytorch_model_weights.pth"))
model.eval()
model = CustomModel(model)
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model_weights/pytorch_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

