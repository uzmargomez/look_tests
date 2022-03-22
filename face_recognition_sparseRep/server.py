import cv2
import base64
import argparse
import numpy as np
from flask import Flask, Response, request, jsonify

from lib.dataset_functions import DataSet
from lib.misc_functions import get_logger
from lib.image_resizer import resize_images

logger = get_logger()

app = Flask(__name__)

resize_images()


@app.route("/", methods=["POST"])
def index():
    data = request.get_json()

    src = data["image"]
    img_decoded = base64.standard_b64decode(src)
    img_buffer = np.frombuffer(img_decoded, np.uint8)
    img = cv2.imdecode(img_buffer, 1)

    res = DS.classify(img)
    classes = DS.classes

    if res == -1:
        class_name = "Not in DS"
    elif res == None:
        class_name = "No detection"
    elif res != -1:
        class_name = classes[res]

    return jsonify({"class": res, "class_name": class_name})


@app.route("/info", methods=["GET"])
def info():
    return jsonify(DS.classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        "-dir",
        help="Dataset directory",
        type=str,
        default="datasets/LookDataSet",
    )
    parser.add_argument(
        "--extension",
        "-ext",
        help="Dataset images extension",
        type=str,
        default="jpg",
    )
    parser.add_argument(
        "--images_per_class",
        "-ipc",
        help="Images to use per class",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--size", "-si", help="Image size", type=int, default=24
    )
    parser.add_argument(
        "--vertical", "-ve", help="Vertical splits", type=int, default=4
    )
    parser.add_argument(
        "--horizontal", "-ho", help="Horizontal splits", type=int, default=2
    )
    parser.add_argument(
        "--epsilon", "-e", help="Epsilon", type=float, default=0.0
    )
    parser.add_argument(
        "--threshold",
        "-t",
        help="Classification threshold",
        type=float,
        default=0.22,
    )
    parser.add_argument(
        "--vis",
        "-v",
        help="Show aligned and crop images",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    DS = DataSet(
        dir=args.directory,
        ext=args.extension,
        images_per_class=args.images_per_class,
        size=args.size,
        vertical=args.vertical,
        horizontal=args.horizontal,
        epsilon=args.epsilon,
        threshold=args.threshold,
        vis=args.vis,
    )

    app.run(
        host="0.0.0.0",
        port="8000",
        threaded=True,
        debug=True,
        use_reloader=False,
    )
