from typing import List, Tuple

import numpy as np
import onnxruntime as rt
from numpy.linalg import norm
from PIL import Image

from crystal_eyes.utils import load_pickle, preprocess


class Classifer:
    HAED_SIZE = 128

    def __init__(self, embedder_p: str, clf_p: str, trans_p: str) -> None:
        super().__init__()
        self.embedder = rt.InferenceSession(embedder_p)
        self.clf = rt.InferenceSession(clf_p)
        self.trans = load_pickle(trans_p)

    def get_feature_vn(self, head0: Image.Image):
        input_ = preprocess(head0, self.HAED_SIZE, torch_nor=True)
        feature_v = self.embedder.run(None, {"input0": input_})[0]  # batch_size 1
        feature_vn = feature_v / np.expand_dims(norm(feature_v, axis=1), axis=1)
        return feature_vn

    def classify(self, feature_vn: np.array, topN=3):
        probs = self.clf.run(None, {"input_1:0": feature_vn})[0]
        top_preds = np.argsort(probs)[:, -topN:][:, ::-1]
        top_probs = [prob[pred] for prob, pred in zip(probs, top_preds)]
        return top_preds, top_probs

    def translate(self, top_preds: np.array, top_probs: np.array):
        ret: List[Tuple[str, float]] = []
        for top_pred, top_prob in zip(top_preds, top_probs):
            ret.extend(
                (self.trans[pred], float(prob))
                for pred, prob in zip(top_pred, top_prob)
            )
        return ret

    def predict(self, head0: Image.Image, topN=3):
        feature_vn = self.get_feature_vn(head0)
        top_preds, top_probs = self.classify(feature_vn, topN=topN)
        return self.translate(top_preds, top_probs)

    def __call__(self, head0: Image.Image, topN=3):
        return self.predict(head0, topN=topN)
