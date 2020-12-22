import json
import os

from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

from crystal_eyes.classifer import Classifer
from crystal_eyes.detector import Detector
from crystal_eyes.utils import pil_loader

app = FastAPI()
clf = None
detector = None

origins = json.loads(
    os.getenv("origins", '["http://localhost:8080", "https://moemoemoe.web.app"]')
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder_p = os.getenv("encoder", "models/encoder.onnx")
tf_model_p = os.getenv("tf_model", "models/tf_model.onnx")
idx_label_p = os.getenv("idx_label", "models/idx_label.pkl")
detector_p = os.getenv("detector", "models/best.quant.onnx")


@app.post("/dectect/")
async def dectect(topN: int = 3, file: bytes = File(...)):
    global clf
    global detector
    if clf is None:
        clf = Classifer(encoder_p, tf_model_p, idx_label_p)
    if detector is None:
        detector = Detector(detector_p, head_clf=clf)
    im0 = pil_loader(file, from_byte=True)
    preds, head_locs = detector.detect(im0)
    return dict(preds=preds, head_locs=head_locs)
