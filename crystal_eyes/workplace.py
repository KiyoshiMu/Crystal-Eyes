from crystal_eyes.classifer import Classifer
from crystal_eyes.detector import Detector
from crystal_eyes.displayer import Displayer
from crystal_eyes.utils import pil_loader


class WorkPlace:
    def __init__(self, detector: Detector, displayer: Displayer = None) -> None:
        self.detector = detector
        self.displayer = displayer

    def take(self, img_p, display=False):
        im0 = pil_loader(img_p)
        preds, head_locs = self.detector.detect(im0)
        if display:
            marked = im0
            for pred, head_loc in zip(preds, head_locs):
                marked = self.displayer.mark(marked, pred, head_loc)
            self.displayer.show(marked)
        else:
            return preds, head_locs


if __name__ == "__main__":
    clf = Classifer(
        "models/encoder.onnx", "models/tf_model.onnx", "models/idx_label.pkl"
    )
    detector = Detector("models/best.quant.onnx", clf)
    wp = WorkPlace(detector)
    wp.take("/mnt/e/Project/moeL/3.jpg")
