#!/usr/bin/env python3
"""FastAPI server for medicinal plant classification - Production Ready for Render"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Architecture
def total_variation_loss(x):
    loss_h = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]))
    loss_w = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]))
    return loss_h + loss_w

class EdgeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,16,3,2,1,bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(16,16,3,2,1,bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        tv = total_variation_loss(x)
        return x, tv

class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,32,1,2,bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(32,3,3,2,1,bias=False), nn.BatchNorm2d(3), nn.LeakyReLU(0.2))
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        tv = total_variation_loss(x)
        return x, tv

class VeinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,32,7,2,3,bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(32,32,5,2,2,bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(32,16,3,1,1,bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.2))
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        tv = total_variation_loss(x)
        return x, tv

class MixerBlock(nn.Module):
    def __init__(self,num_tokens,embed_dim,token_dim=128,channel_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mixing = nn.Sequential(nn.Linear(num_tokens,token_dim), nn.GELU(), nn.Linear(token_dim,num_tokens))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mixing = nn.Sequential(nn.Linear(embed_dim,channel_dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(channel_dim,embed_dim))
    def forward(self,x):
        y = self.norm1(x)
        y = y.transpose(1,2)
        y = self.token_mixing(y)
        y = y.transpose(1,2)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y
        return x

class VECTVMixer(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.edge = EdgeBranch()
        self.color = ColorBranch()
        self.vein = VeinBranch()
        self.fusion = nn.Sequential(nn.Conv2d(35,48,1,bias=False), nn.BatchNorm2d(48), nn.LeakyReLU(0.2))
        self.constituent = nn.Sequential(nn.Conv2d(48,64,1,bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.down1 = nn.Sequential(nn.Conv2d(64,96,3,2,1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(96,128,3,2,1), nn.BatchNorm2d(128))
        self.num_tokens = 14*14
        self.mixer = nn.Sequential(*[MixerBlock(self.num_tokens,128) for _ in range(4)])
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128,28)
        self.bn = nn.BatchNorm1d(28)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(28,num_classes)

    def forward(self,x):
        e,tv1 = self.edge(x)
        c,tv2 = self.color(x)
        v,tv3 = self.vein(x)
        x = torch.cat([e,c,v],dim=1)
        x = self.fusion(x)
        x = self.constituent(x)
        x = self.down1(x)
        x = self.down2(x)
        x = x.flatten(2).transpose(1,2)
        x = self.mixer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        tv_loss = 1e-4*tv1 + 5e-5*tv2 + 1e-4*tv3
        return logits, tv_loss

# Initialize FastAPI
app = FastAPI(
    title="Medicinal Plant Classifier API",
    description="Classify medicinal plants using VECTVMixer model",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and labels
logger.info("🔄 Loading model...")
device = torch.device('cpu')
model = VECTVMixer(num_classes=10)

model_path = 'best_vectvmixer.pth'
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"✅ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        sys.exit(1)
else:
    logger.error(f"❌ Model file not found: {model_path}")
    sys.exit(1)

model.to(device)
model.eval()

# Load labels
labels_path = 'labels.txt'
labels = []
if os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    logger.info(f"✅ Loaded {len(labels)} plant labels")
else:
    logger.error(f"❌ Labels file not found: {labels_path}")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

logger.info("✅ Server ready!")

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

@app.get("/")
async def root():
    return {
        "name": "Medicinal Plant Classifier",
        "description": "API for plant classification",
        "version": "1.0.0",
        "endpoints": {"/health", "/predict"}
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "VECTVMixer",
        "classes": len(labels),
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not labels:
            raise HTTPException(status_code=500, detail="Model labels not loaded")

        contents = await file.read()
        image = Image.open(BytesIO(contents))
        logger.info(f"📸 Received image: {file.filename}")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(img_tensor)

        logits_np = logits.cpu().numpy().flatten()
        pred_idx = np.argmax(logits_np)
        probs = softmax(logits_np)

        return {
            "success": True,
            "predicted_class": labels[pred_idx] if pred_idx < len(labels) else "Unknown",
            "confidence": float(probs[pred_idx]),
            "all_predictions": {labels[i]: float(probs[i]) for i in range(len(labels))}
        }
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
