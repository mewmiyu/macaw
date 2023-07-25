import os

import torchvision.transforms as transforms
import numpy as np
import yaml

from PIL import Image

import argparse
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream

import cv2 as cv
import pickle
import imutils


def vid_handler(file):
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default=file, help="path to input video file")
    args = vars(ap.parse_args())

    return FileVideoStream(args["video"], queue_size=128).start()


def webcam_handler():
    return WebcamVideoStream(src=0).start()


def save_descriptor_to_file(file, data):
    pickle.dump(file, data)


def load_descriptor_from_file(file):
    return pickle.load(file)


def load_img(filename: str, size: tuple = None) -> tuple[np.ndarray, np.ndarray]:
    img = cv.imread(filename)
    if size:
        img = resize(img, size[1])
        # scale_percent = int(100 * size[0] / img.shape[0])
        # scale_percent = max(scale_percent, int(100 * size[1] / img.shape[1]))
        #
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        #
        # img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return img, gray


def crop_img(
    img: np.ndarray, min_y: int, min_x: int, max_y: int, max_x: int
) -> np.ndarray:
    return img[min_x:max_x, min_y:max_y, :]


def resize(img, width):
    return imutils.resize(img, width=width)


def to_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def gen_triplet_dataset(labels, batch_size, batch_amount):
    """
    This function generates a dataset based on triplets. It returns
    a numpy array of size [batch_amount, batch_size, 3]. Each entry
    is an index describing the position of the data based on the
    labels input
    """
    dataset = []
    max_label = np.max(labels)
    for _ in range(batch_amount):
        batch = []
        for b in range(batch_size):
            label1 = np.random.randint(0, max_label + 1)
            label2 = np.random.randint(0, max_label + 1)
            while label1 == label2:
                label2 = np.random.randint(0, max_label + 1)
            label1_pos = np.where(labels == label1)
            l1_min = np.min(label1_pos)
            l1_max = np.max(label1_pos)
            label2_pos = np.where(labels == label2)
            l2_min = np.min(label2_pos)
            l2_max = np.max(label2_pos)
            anchor = np.random.randint(l1_min, l1_max + 1)
            positive = np.random.randint(l1_min, l1_max + 1)
            while positive == anchor and l1_min != l1_max:
                positive = np.random.randint(l1_min, l1_max + 1)
            negative = np.random.randint(l2_min, l2_max + 1)
            batch.append([anchor, positive, negative])
        dataset.append(np.array(batch))
    return np.array(dataset)


def read_yaml(filepath):
    """
    Reads in a yaml config file from a filepath
    """
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)
    return data
