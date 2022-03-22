import cv2
import logging
import numpy as np
import mxnet as mx
import os
import time

from lib.mtcnn_detector import MtcnnDetector

logging.basicConfig(level=logging.INFO)

detector = MtcnnDetector(
    model_folder="lib/models", ctx=None, num_worker=4, accurate_landmark=False
)


def detect_and_align(im, size=24, vis=False):
    if isinstance(im, str):
        try:
            original = cv2.imread(im, 1)
        except:
            logging.error("Image type not supported")
            return None
    elif isinstance(im, np.ndarray):
        original = im

    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    detections = detector.detect_face(original)

    if detections is None:
        return None  # Return None if no face is found

    boxes = detections[0]
    points = detections[1]

    chips = detector.extract_image_chips(image, points)

    aligned = chips[0]

    b = boxes[0]
    p = points[0]

    aligned_resized = cv2.resize(
        aligned, (size, size), interpolation=cv2.INTER_AREA
    )

    if vis:
        show = cv2.rectangle(
            original,
            (int(b[0]), int(b[1])),
            (int(b[2]), int(b[3])),
            (0, 255, 0),
            2,
        )
        for i in range(5):
            show = cv2.circle(show, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)
        cv2.imshow("box", show)
        cv2.waitKey(1)

    if vis:
        cv2.imshow("aligned resized", aligned_resized)
        cv2.waitKey(1)

    if vis:
        cv2.imshow("crop", aligned)
        cv2.waitKey(1)

    return aligned_resized
