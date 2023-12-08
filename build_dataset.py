from imutils import paths
from config import *
import os
import shutil

imagePaths = list(paths.list_images("dataset/original"))

BASE_PATH = "dataset/splitted"

VAL = int(len(imagePaths) * TRAIN_SPLIT )
TRAIN = int(len(imagePaths) * TRAIN_SPLIT * (1 - VAL_SPLIT))
datasets = [
    ("train", imagePaths[:VAL], TRAIN_PATH),
    ("val", imagePaths[TRAIN:VAL], VAL_PATH),
    ("test", imagePaths[TRAIN:], TEST_PATH),
]

for dstype, inputPaths, baseOutput in datasets:
    if not os.path.exists(baseOutput):
        os.mkdir(baseOutput)
    for imagePath in inputPaths:
        file_name = imagePath.split(os.path.sep)[-1]
        label = file_name[-5:-4]
        dir_path = os.path.join(baseOutput, label)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        target_dest = os.path.join(dir_path, file_name)
        shutil.copy2(imagePath, target_dest)
