import io
import json
import os
import pickle

import numpy as np
from PIL import Image

MEAN = [0.6826, 0.6033, 0.5841]
STD = [0.2460, 0.2537, 0.2448]
HAED_SIZE = 128
IMG_SIZE = 640


def dump_chara2json(
    dst="chara.json", idx_label_p=os.getenv("idx_label", "models/idx_label.pkl")
):
    with open(dst, "w", encoding="utf8") as t:
        charas = load_pickle(idx_label_p)
        json.dump(list(charas), t)


def load_pickle(path):
    with open(path, "rb") as t:
        return pickle.load(t)


def pil_loader(src, from_byte=False):
    if from_byte:
        src = io.BytesIO(src)
    im_rgb = Image.open(src).convert("RGB")
    return im_rgb


def preprocess(img: Image.Image, size, torch_nor=True):
    im = np.array(img.resize((size, size))) / 255
    if torch_nor:
        im = (im - MEAN) / STD
    im = np.expand_dims(im.transpose(2, 0, 1), axis=0)
    return im.astype(np.float32)
