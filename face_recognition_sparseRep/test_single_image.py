import cv2
import random
import argparse
import sys
import datetime

from lib.Pipeline import detector
from lib.dataset_functions import DataSet
from lib.misc_functions import get_logger


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
    default=14,
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
    default=0.1,
)
parser.add_argument(
    "--vis",
    "-v",
    help="Show aligned and crop images",
    type=bool,
    default=False,
)
args = parser.parse_args()

ds = DataSet(
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

print(ds.test_images_known)

ds.classify('datasets/LookDataSet/Test/Salma_Hayek/Salma_Hayek_017_resized.jpg', plot=True)