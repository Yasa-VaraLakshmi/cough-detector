from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import DEFAULT_SR, build_models, extract_features, load_audio_bytes, load_models


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="Cough Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_model_bundle = None
COUGH_THRESHOLD = 0.2


@app.on_event("startup")
def _startup() -> None:
    global _model_bundle
    _model_bundle = load_models(MODELS_DIR)
    if _model_bundle is None:
        _model_bundle = build_models(DATA_DIR, MODELS_DIR)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = build_models(DATA_DIR, MODELS_DIR)

    audio_bytes = await file.read()
    y, sr = load_audio_bytes(audio_bytes, target_sr=DEFAULT_SR)
    features = extract_features(y, sr).reshape(1, -1)

    cough_proba = _model_bundle.cough_model.predict_proba(features)[0]
    cough_classes = _model_bundle.cough_labels.inverse_transform(
        list(range(len(cough_proba)))
    )
    cough_map = {label: float(prob) for label, prob in zip(cough_classes, cough_proba)}

    cough_prob = cough_map.get("cough", 0.0)
    cough_present = cough_prob >= COUGH_THRESHOLD
    cough_label = "cough" if cough_present else "non_cough"

    return JSONResponse(
        {
            "cough_label": cough_label,
            "cough_present": cough_present,
        }
    )

@app.post("/api/refresh")
def refresh_models():
    global _model_bundle
    _model_bundle = build_models(DATA_DIR, MODELS_DIR)
    return JSONResponse({"status": "ok"})
