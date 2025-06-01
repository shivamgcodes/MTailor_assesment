# check if     inp = mtailor.preprocess_numpy(img).unsqueeze(0) 
# is same as preprocessed input from my model.


#check the preprocessing of the input image with my class and their class
#uske baad make the onnx model and then check the outputs for both models
#chek with some batch size
#test model loading, and inference on GPU

import torch
import numpy as np
import cv2
from pytorch_model import Classifier, BasicBlock
from CustomModel import CustomModel
import torch.nn.functional as F
from PIL import Image

import torch
import numpy as np
import cv2
from pytorch_model import Classifier, BasicBlock
from CustomModel import CustomModel
import torch.nn.functional as F
from PIL import Image
import os
from torchvision.utils import save_image
import onnxruntime as ort



# Load model
original_model = Classifier(BasicBlock, [2, 2, 2, 2])
original_model.load_state_dict(torch.load("model_weights/pytorch_model_weights.pth"))
original_model.eval()
custom_model = CustomModel(original_model)

img = Image.open("sample_images/n01440764_tench.jpeg").convert("RGB")
cv2_img = np.array(img)  # Already RGB now, shape: (H, W, 3)

original_preprocess = original_model.preprocess_numpy(img).unsqueeze(0)  # Output: torch.Tensor

x = torch.tensor(cv2_img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0  # Shape: [1, 3, H, W], float32
custom_preprocess = custom_model.preprocess_torch(x)

# 🔍 Compare both preprocessing outputs
print("Original Preprocess Shape:", original_preprocess.shape)
print("Custom Preprocess Shape:", custom_preprocess.shape)
mse = torch.mean((original_preprocess - custom_preprocess)**2).item()
print("MSE between outputs:", mse)

os.makedirs("preprocessed_outputs", exist_ok=True)

# Save directly without any unnormalization
save_image(original_preprocess, "preprocessed_outputs/original_preprocess.jpg")
save_image(custom_preprocess, "preprocessed_outputs/custom_preprocess.jpg")


session = ort.InferenceSession("model_weights/pytorch_model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
actual_provider = session.get_providers()[0]
print("Execution providers:", session.get_providers())

# Get input & output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
outputs = session.run([output_name], {input_name: x.detach().cpu().numpy()})
preds = outputs[0]

print("Model output:", preds.argmax(axis=1))

inp = original_model.preprocess_numpy(img).unsqueeze(0) 
res = original_model.forward(inp)
print("Original model output:", res.argmax())


# x is a single input: shape [1, 3, 224, 224]
x_batched = x.repeat(4, 1, 1, 1)  # Now shape is [4, 3, 224, 224]

print("Batched input shape:", x_batched.shape)

# Run inference on batched input
outputs = session.run([output_name], {input_name: x_batched.detach().cpu().numpy()})
preds = outputs[0]

print("Model outputs shape:", preds.shape)
print("Predicted classes:", preds.argmax(axis=1))

results = {
    "CUDA_Used": actual_provider == "CUDAExecutionProvider",
    "MSE_OK": mse < 3e-3,
    "Shape_Match": preds.shape[0] == len(x_batched),
    "Prediction_Match": preds.argmax(axis=1)[0] == res.argmax().item()
}


assert mse < 3e-3, "Preprocessing outputs do not match!" 
assert preds.argmax(axis=1)[0] == res.argmax().item(), "Mismatch between ONNX and original PyTorch model outputs"
assert actual_provider == "CUDAExecutionProvider", "ONNX Runtime is not using CUDA!"
assert preds.shape[0] == len(x_batched), "Batch size mismatch in ONNX model output!"
