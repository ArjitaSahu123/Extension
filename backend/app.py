#!/usr/bin/env python3
"""
app.py - Flask backend for AI vs Fake detector.
Robust model loading: supports common checkpoint formats and both single-logit (1) and 2-class heads.
Returns JSON: {"prediction": {"label": "real"|"fake", "probability": 0.XX}}

Place your model file at either:
 - ./models/classifier_resnet50.pth
 - /mnt/data/classifier_resnet50.pth

Requires: Flask, torchvision, torch, Pillow, requests, flask_cors
"""
import io
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np    

# ----- Config -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Try several likely paths for the model
MODEL_DIR = Path(__file__).parent / "models"
FALLBACK_MODEL_PATH = Path("/mnt/data/classifier_resnet50.pth")
MODEL_PATH = MODEL_DIR / "classifier_resnet50.pth"
if not MODEL_PATH.exists() and FALLBACK_MODEL_PATH.exists():
    MODEL_PATH = FALLBACK_MODEL_PATH

# Max size to download in bytes (10 MB)
MAX_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT = 15

# ----- Logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_vs_fake_app")

# ----- Flask app -----
app = Flask(__name__)
CORS(app)

# ----- Transforms -----
# Default to ImageNet normalization + center crop/resize commonly used in ResNet training.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----- Model building helper -----
def get_resnet50_model(num_classes: int = 2) -> nn.Module:
    """
    Build a ResNet50 and replace the final fc layer to `num_classes`.
    Use pretrained=False here because we'll load our own weights.
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# We'll attempt to load with num_classes=2 first, but prediction code supports 1 or 2 outputs.
model: Optional[nn.Module] = None
model_loaded = False

def _remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix if present (from DataParallel)."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state

def try_load_model(path: Path) -> (Optional[nn.Module], bool):
    """
    Try several common loading patterns. Returns (model, loaded_flag).
    Loads into a ResNet50 with num_classes=2 by default and tries non-strict load.
    """
    if not path.exists():
        logger.warning(f"Model file not found at {path}")
        return None, False

    # First build a 2-class model (most common)
    candidate = get_resnet50_model(num_classes=2).to(DEVICE)

    try:
        ckpt = torch.load(str(path), map_location=DEVICE)
    except Exception as e:
        logger.exception(f"Failed to load checkpoint file: {e}")
        return None, False

    # Determine whether ckpt is dict or state_dict
    if isinstance(ckpt, dict):
        # Common container keys
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # Could be a bare state_dict (mapping of tensor values)
            # or a mixed dict
            # Heuristic: if many values are tensors => treat as state_dict
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state = ckpt
            else:
                # Unknown dict structure
                logger.warning(f"Checkpoint dict keys: {list(ckpt.keys())}")
                # Attempt to find a nested state_dict-like entry
                found = None
                for k, v in ckpt.items():
                    if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                        found = v
                        break
                if found is None:
                    logger.error("Could not locate a state_dict inside checkpoint.")
                    return None, False
                state = found
    else:
        # If checkpoint is not a dict (unlikely), fail.
        logger.error("Unrecognized checkpoint format (not a dict).")
        return None, False

    # Remove module prefixes if present
    state = _remove_module_prefix(state)

    # Attempt to load into 2-class model non-strictly and report mismatches
    try:
        load_result = candidate.load_state_dict(state, strict=False)
        # load_result is an object with missing_keys/unexpected_keys in recent PyTorch,
        # but torch returns NamedTuple or raises. We'll just log the load_result.
        logger.info(f"Loaded checkpoint to ResNet50(num_classes=2). Non-strict load result: {load_result}")
        candidate.eval()
        return candidate, True
    except Exception as e:
        logger.exception(f"Failed to load state_dict into 2-class model: {e}")

    # As a fallback, try building a 1-output model (single logit -> sigmoid)
    try:
        candidate1 = get_resnet50_model(num_classes=1).to(DEVICE)
        state_for_1 = state.copy()
        # Attempt load non-strict
        load_result = candidate1.load_state_dict(state_for_1, strict=False)
        logger.info(f"Loaded checkpoint to ResNet50(num_classes=1). Non-strict load result: {load_result}")
        candidate1.eval()
        return candidate1, True
    except Exception as e:
        logger.exception(f"Failed to load state_dict into 1-output model: {e}")

    logger.error("All model loading attempts failed.")
    return None, False

# Try to load model at startup
model, model_loaded = try_load_model(MODEL_PATH)
if model_loaded:
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
else:
    logger.warning("Model not loaded. Prediction endpoints will return an error until a valid model is available.")

# ----- Image fetching helper -----
def fetch_image_from_url(url: str) -> Image.Image:
    """
    Download image bytes from a URL and return a PIL Image object.
    Raises exception on failure.
    """
    logger.info(f"Fetching image from URL: {url}")
    headers = {"User-Agent": "ai-vs-fake-bot/1.0"}
    resp = requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    total = 0
    data = io.BytesIO()
    for chunk in resp.iter_content(1024):
        if chunk:
            data.write(chunk)
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError(f"Image too large ({total} bytes). Limit is {MAX_BYTES} bytes.")
    data.seek(0)
    img = Image.open(data).convert("RGB")
    return img

# ----- Prediction helper -----
def predict_pil_image(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Runs the model on a PIL image and returns {'label': 'real'/'fake', 'probability': float}
    Supports both 1-output (sigmoid) and 2-output (softmax) model heads.
    """
    if not model:
        raise RuntimeError("Model is not loaded on the server.")

    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)  # shape [1, C, H, W]
    with torch.no_grad():
        out = model(tensor)
        # Ensure out is a tensor
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out)

        out = out.cpu()
        if out.ndim == 2 and out.shape[1] == 1:
            # shape [1,1] -> single logit
            prob = float(torch.sigmoid(out.squeeze(1)).numpy()[0])
            label = "real" if prob >= 0.5 else "fake"
        elif out.ndim == 2 and out.shape[1] == 2:
            probs = torch.softmax(out, dim=1).numpy()[0]
            cls = int(probs.argmax())
            prob = float(probs[cls])
            # Assume training mapping: class 1 => real, class 0 => fake
            label = "real" if cls == 1 else "fake"
        else:
            # Unexpected shape: try treating as single value
            out_flat = out.flatten().numpy()
            if out_flat.size == 1:
                prob = float(1.0 / (1.0 + np.exp(-out_flat[0])))
                label = "real" if prob >= 0.5 else "fake"
            else:
                raise RuntimeError(f"Unexpected model output shape: {out.shape}")

    return {"label": label, "probability": float(round(prob, 6))}

# ----- Routes -----
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(model_loaded),
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH)
    })

@app.route("/predict-url", methods=["POST"])
def predict_url():
    """
    Expects JSON: { "imageUrl": "<url>" }
    Returns: { "prediction": { "label": "real"|"fake", "probability": 0.XX } }
    """
    if not model_loaded:
        return jsonify({"error": "model_not_loaded", "message": "Model not loaded on server."}), 500

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "bad_request", "message": "Expected JSON body."}), 400

    image_url = body.get("imageUrl") or body.get("imageURL") or body.get("url")
    if not image_url:
        return jsonify({"error": "bad_request", "message": "Expected 'imageUrl' in JSON body."}), 400

    try:
        pil_image = fetch_image_from_url(image_url)
        result = predict_pil_image(pil_image)
        return jsonify({"prediction": result})
    except Exception as e:
        logger.exception(f"Error in /predict-url: {e}")
        return jsonify({"error": "prediction_failed", "message": str(e)}), 500

@app.route("/predict-b64", methods=["POST"])
def predict_b64():
    """
    Expects JSON: { "imageBase64": "<base64 data URI or raw base64>" }
    Returns same as /predict-url
    """
    if not model_loaded:
        return jsonify({"error": "model_not_loaded", "message": "Model not loaded on server."}), 500

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "bad_request", "message": "Expected JSON body."}), 400

    b64 = body.get("imageBase64") or body.get("b64")
    if not b64:
        return jsonify({"error": "bad_request", "message": "Expected 'imageBase64' in JSON body."}), 400

    # Normalize data URI if present
    if b64.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        try:
            _, payload = b64.split(",", 1)
            b64 = payload
        except Exception:
            return jsonify({"error": "bad_request", "message": "Invalid data URI format."}), 400

    import base64
    try:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = predict_pil_image(img)
        return jsonify({"prediction": result})
    except Exception as e:
        logger.exception(f"Error decoding base64 image: {e}")
        return jsonify({"error": "prediction_failed", "message": str(e)}), 500

# ----- Run server -----
if __name__ == "__main__":
    if not MODEL_DIR.exists():
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

