import os
import cv2
import glob

MAX_DIM = 400
DATASET = "datasets/**"
EXT = "jpg"
VIS = False

all_dirs = glob.glob(DATASET,recursive=True)
for i_dir, dir in enumerate(all_dirs):
    all_paths = glob.glob(dir + "/*." + EXT)
    for i_path, path in enumerate(all_paths):
        print("Resizing directories {:<3}/{:<3} Resizing images {:<3}/{:<3}".format(i_dir+1, len(all_dirs), i_path+1, len(all_paths)), end="\r")
        image = cv2.imread(path, 1)
        height = image.shape[0]
        width = image.shape[1]
        if height > MAX_DIM or width > MAX_DIM:
            factor = max((width, height))/MAX_DIM
            new_width = int(width/factor)
            new_height = int(height/factor)
            new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            os.remove(path)
            splitext = list(os.path.splitext(path))
            splitext.insert(-1, "_resized")
            new_path = "".join(splitext)
            cv2.imwrite(new_path, new_image)

            if VIS:
                cv2.imshow("image", image)
                cv2.imshow("new_image", new_image)
                cv2.waitKey(1)

print("\nFinished resizing")
