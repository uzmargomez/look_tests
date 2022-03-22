import cv2
import random
import argparse
import sys
import datetime

from lib.Pipeline import detector
from lib.dataset_functions import DataSet
from lib.misc_functions import get_logger

logger = get_logger(logname='log.txt')

random.seed(13)


def main():

    text_file = open('output.txt', 'w')

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
        "--size", "-si", help="Image size", type=int, default=56
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

    vis = args.vis

    counter = 0
    i = 0
    begin_time = datetime.datetime.now()
    for subject in ds.test_images_known:
        for image in ds.test_images_known[subject]:
            test_res = ds.classify(image, vis=vis)
            counter = counter + 1
            if test_res == -1:
                result = "incorrect"
                text_file.write(
                    "{:<30}| {:<30}| {:<10}\n".format(
                        subject, "* NOT IN DB *", result
                    )
                )
            elif test_res is None:
                text_file.write("{:<30}| {:<30}|\n".format(subject, "* NO FACE FOUND *"))
            else:
                if ds.classes[test_res] == subject:
                    result = "correct"
                    i = i + 1
                else:
                    result = "incorrect"
                text_file.write(
                    "{:<30}| {:<30}| {:<10}\n".format(
                        subject, ds.classes[test_res], result
                    )
                )

    text_file.close()


if __name__ == "__main__":
    main()
