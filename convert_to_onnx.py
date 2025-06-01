from pytorch_model import Classifier, BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomModel import CustomModel

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

