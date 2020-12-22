import json

import cv2
import matplotlib.pyplot as plt


class Displayer:
    def __init__(self) -> None:
        super().__init__()

    def mark(self, im0, pred, head_loc):
        img = self._to_width(im0)
        h, w, _ = img.shape
        h0, w0, _ = im0.shape
        hg = h / h0
        wg = w / w0
        ((lx, dy), (rx, ty)) = self._adjust_headloc(head_loc, wg, hg)
        color = (255, 0, 0)
        imDraw = cv2.rectangle(img, (lx, dy), (rx, ty), color, 3)
        text = json.dumps(pred, indent=2).replace("{", "").replace("}", "")
        lines = text.split("\n")[1:4]
        marked = self.put_text(imDraw, lines, lx, ty)
        return marked

    def _to_width(self, img, new_w=640):
        h, w, _ = img.shape
        new_h = int(new_w / w * h)
        return cv2.resize(img, (new_w, new_h))

    def _adjust_headloc(self, head_loc, wg, hg):
        ((lx, dy), (rx, ty)) = head_loc
        return ((int(lx * wg), int(dy * hg)), (int(rx * wg), int(ty * hg)))

    def show(self, img, label=""):
        plt.figure(dpi=150)
        plt.imshow(img)
        plt.axis("off")

    def put_text(self, img, lines, text_offset_x, text_offset_y):
        font_scale = 1
        font = cv2.FONT_HERSHEY_PLAIN
        # set the rectangle background to white
        rectangle_bgr = (255, 255, 255)
        # get the width and height of the text box

        sizes = [
            cv2.getTextSize(line, font, fontScale=font_scale, thickness=1)[0]
            for line in lines
        ]
        box_width = max(s[0] for s in sizes)

        text_height = sizes[0][1]
        padding = int(text_height * 0.8)

        box_height = sum(s[1] for s in sizes) + (len(sizes) - 1) * padding
        box_coords = (
            (text_offset_x, text_offset_y),
            (text_offset_x + box_width + 2, text_offset_y + box_height + padding),
        )
        text_offset_y += text_height + 10
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        for i, line in enumerate(lines):
            # make the coords of the box with a small padding of two pixels
            y = text_offset_y + i * (text_height + padding)
            cv2.putText(
                img,
                line,
                (text_offset_x, y),
                font,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=1,
            )
        return img
