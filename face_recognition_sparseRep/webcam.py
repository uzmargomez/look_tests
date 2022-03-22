import os
import cv2
import json
import base64
import requests
import numpy as np
import mxnet as mx
import time
from datetime import datetime

from lib.mtcnn_detector import MtcnnDetector

detector = MtcnnDetector(
    model_folder="lib/models", ctx=None, num_worker=4, accurate_landmark=False
)

video_path = 0
cap = cv2.VideoCapture(video_path)

color = (246, 181, 100)


def post(data, headers):
    """
    Función para realizar peticiones a la API de reconocimiento facial
    """
    try:
        r = requests.post(
            "http://localhost:8000/", data=json.dumps(data), headers=headers
        )
        resp = json.loads(r.text)
        return resp
    except ConnectionRefusedError:
        return None


while True:
    ok, frame = cap.read()
    if ok:
        """
        Se realiza detección de rostros
        """
        # img = cv2.resize(frame,(320,240))
        img = frame

        preview = img.copy()

        detections = detector.detect_face(img)

        if detections is None:
            continue  # Return None if no face is found

        boxes = detections[0]
        points = detections[1]

        chips = detector.extract_image_chips(img, points)

        for chip, box in zip(chips, boxes):

            """
            Se codifica en base64 cada rostro encontrado y se envía a la API

            """
            _, enccrop = cv2.imencode(".jpg", chip)
            b64crop = base64.b64encode(enccrop)
            b64crop = b64crop.decode("utf-8")
            data = {"image": b64crop}
            headers = {"content-type": "application/json"}

            id = post(data, headers)

            if id is not None:
                cv2.rectangle(
                    preview,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    1,
                )
                cv2.putText(
                    preview,
                    str(id["class_name"]),
                    (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                max_x = preview.shape[1]
                max_y = preview.shape[0]

                im_boundary = cv2.resize(
                    chip, (24, 24), interpolation=cv2.INTER_AREA
                )

                if max_x - int(box[2]) <= int(box[0]) and max_y - int(
                    box[3]
                ) <= int(box[1]):
                    preview[
                        int(box[1]) - im_boundary.shape[0] : int(box[1]),
                        int(box[0]) - im_boundary.shape[1] : int(box[0]),
                        :,
                    ] = im_boundary

                elif max_x - int(box[2]) >= int(box[0]) and max_y - int(
                    box[3]
                ) >= int(box[1]):
                    preview[
                        int(box[3]) : int(box[3]) + im_boundary.shape[0],
                        int(box[2]) : int(box[2]) + im_boundary.shape[1],
                        :,
                    ] = im_boundary

                elif max_x - int(box[2]) <= int(box[0]) and max_y - int(
                    box[3]
                ) >= int(box[1]):
                    preview[
                        int(box[3]) : int(box[3]) + im_boundary.shape[0],
                        int(box[0]) - im_boundary.shape[1] : int(box[0]),
                        :,
                    ] = im_boundary

                elif max_x - int(box[2]) >= int(box[0]) and max_y - int(
                    box[3]
                ) <= int(box[1]):
                    preview[
                        int(box[1]) - im_boundary.shape[0] : int(box[1]),
                        int(box[2]) : int(box[2]) + im_boundary.shape[1],
                        :,
                    ] = im_boundary

                print(id["class"])

        cv2.imshow("preview", preview)

        k = cv2.waitKey(0)
        if k == 27:  # Esc key to stop
            break
