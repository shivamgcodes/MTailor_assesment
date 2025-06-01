# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2
import numpy as np
import torch
import onnxruntime as ort

session = ort.InferenceSession("model_weights/pytorch_model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Image must be JPEG or PNG")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv2_img = np.array(image)  # Already RGB now, shape: (H, W, 3)

    x = torch.tensor(cv2_img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0  # Shape: [1, 3, H, W], float32
        # Get input & output names


    # Run inference
    outputs = session.run([output_name], {input_name: x.detach().cpu().numpy()})
    preds = outputs[0]
    predicted_class = preds.argmax(axis=1)

    return JSONResponse({"predicted class": predicted_class.item()})


@app.get("/health")
def health():
    return "OK"

@app.get("/test")
def test():
    from test import results
    return JSONResponse({"results": results})