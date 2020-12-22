import os

import pytest

from crystal_eyes.classifer import Classifer
from crystal_eyes.detector import Detector, HeadClf
from crystal_eyes.utils import pil_loader

encoder_p = os.getenv("encoder", "models/encoder.onnx")
tf_model_p = os.getenv("tf_model", "models/tf_model.onnx")
idx_label_p = os.getenv("idx_label", "models/idx_label.pkl")
detector_p = os.getenv("detector", "models/best.quant.onnx")


@pytest.fixture()
def classifer():
    clf = Classifer(encoder_p, tf_model_p, idx_label_p)
    return clf


@pytest.fixture()
def detector(classifer):
    detector = Detector(detector_p, classifer)
    return detector


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (
            "data/head0.jpg",
            "abe_nana",
        ),
        (
            "data/head1.jpg",
            "c.c.",
        ),
    ],
)
def test_classifer(classifer: HeadClf, inputs, expected):
    head0 = pil_loader(inputs, from_byte=False)
    preds = classifer.predict(head0, 3)
    pred_names = [pred[0] for pred in preds]
    assert expected in pred_names and expected == pred_names[0]


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (
            "data/test0.jpg",
            "rem_(re:zero)",
        ),
        (
            "data/test1.jpg",
            "misaka_mikoto",
        ),
    ],
)
def test_detector(detector, inputs, expected):
    im0 = pil_loader(inputs, from_byte=False)
    preds, _ = detector.detect(im0)
    pred_names = [pred[0] for pred in preds[0]]  # only one head
    assert expected in pred_names
