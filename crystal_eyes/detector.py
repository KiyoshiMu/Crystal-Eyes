import logging
from abc import abstractmethod
from typing import Protocol

import numpy as np
import onnxruntime as rt
from PIL import Image

from crystal_eyes.utils import pil_loader, preprocess

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HeadClf(Protocol):
    @abstractmethod
    def predict(self, head0: Image.Image, topN: int = 3) -> list:
        raise NotImplementedError


class Detector:
    IMG_SIZE = 640

    def __init__(
        self,
        yolo_p: str,
        head_clf: HeadClf,
    ) -> None:
        self.yolo = rt.InferenceSession(yolo_p)
        self.head_clf = head_clf

    def get_head_loc(self, im0: Image.Image, prob_thresh=0.2, overlapThresh=0.5):
        pred_onx = self.yolo.run(
            None, {"images": preprocess(im0, self.IMG_SIZE, torch_nor=False)}
        )[0][0]

        pre_boxes = list(filter(lambda row: row[4] * row[5] > prob_thresh, pred_onx))
        if len(pre_boxes) == 0:
            pre_boxes = [max(pred_onx, key=lambda row: row[4] * row[5])]
        tmp_boxes_ = np.array(list(map(to_boxes, pre_boxes)))
        tmp_boxes = non_max_suppression(tmp_boxes_, overlapThresh)
        h0 = im0.height
        w0 = im0.width
        wg = self.IMG_SIZE / w0
        hg = self.IMG_SIZE / h0
        boxes = list(map(lambda x: box_adjust(x, wg, w0, hg, h0), tmp_boxes))
        return boxes

    def detect(self, im0: Image.Image, topN=3):
        boxes = self.get_head_loc(im0)
        preds = []
        head_locs = []
        for lx, ly, rx, ry in boxes:
            head_loc = ((lx, ly), (rx, ry))
            head0 = im0.crop((lx, ly, rx, ry))
            pred = self.head_clf.predict(head0, topN=topN)
            preds.append(pred)
            head_locs.append(head_loc)
        return preds, head_locs

    def detect_from_path(self, img_p):
        im0 = pil_loader(img_p)
        return self.detect(im0)


def to_boxes(prebox):
    w_half = prebox[2] / 2
    h_half = prebox[3] / 2
    lx = prebox[0] - w_half
    ly = prebox[1] - h_half
    rx = prebox[0] + w_half
    ry = prebox[1] + h_half
    return [lx, ly, rx, ry]


def box_adjust(tmp_box, wg, w0, hg, h0):
    lx = _clip(tmp_box[0] / wg, w0)
    ly = _clip(tmp_box[1] / hg, h0)
    rx = _clip(tmp_box[2] / wg, w0)
    ry = _clip(tmp_box[3] / hg, h0)
    return [int(v) for v in (lx, ly, rx, ry)]


def _clip(v, up, down=0):
    if v < down:
        return down
    if v > up:
        return up
    else:
        return v


# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
