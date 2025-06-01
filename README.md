# MTailor Assessment

This repository contains code for preprocessing, converting a PyTorch model to ONNX, validating the conversion, and deploying via [Cerebrium](https://www.cerebrium.ai/).

---

## 🔧 Setup Instructions

### 1. Check Python 3.10

Ensure Python 3.10 is installed:

```bash
python3 --version
```

If not, install it before proceeding.

### 2. Create and Activate Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📦 Download Model Weights

Run the script to download the model weights:

```bash
python scripts/download_weights.py
```

This will populate the `model_weights/` directory with the required `.pth` file if it doesn't already exist.

---

## 🔁 Convert to ONNX

Convert the PyTorch model to ONNX:

```bash
python convert_to_onnx.py
```

This uses the `.pth` file from `model_weights/` and generates a `.onnx` file in the same directory.

---

## ✅ Validate Conversion

Run the test script to validate preprocessing and ONNX model behavior:

```bash
python test.py
```

### It checks:
- Whether preprocessing between original and converted model is identical  
- Whether the outputs of `.pth` and `.onnx` models match  
- Whether the ONNX model supports batched inference  
- Whether ONNX runs on the **CUDAExecutionProvider** if available  

⚠️ Make sure the `sample_images/` directory contains:

```
sample_images/n01440764_tench.jpeg
```

Or update the path in `test.py` accordingly.

---

## 🚀 Deployment

After validation, deploy the app using:

```bash
cerebrium login
cerebrium deploy
```

---

## 📡 Test the Deployed Endpoint

Use the following `curl` command to test the deployed model:

```bash
curl --location --request POST 'https://api.cortex.cerebrium.ai/v4/<project-id>/<deployment>/predict' \
--header 'Authorization: Bearer <your-token>' \
--form 'file=@sample_images/n01440764_tench.jpeg'
```

> 🔁 Replace `<project-id>`, `<deployment>`, and `<your-token>` accordingly.

> test_server.py only contains the method structure, it is hollow as of now, the logic for the actual testing is yet to be implemented.