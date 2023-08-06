import numpy as np
import yaml
import glob
from pathlib import Path

from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream

import cv2 as cv
import pickle
import imutils
from collections import namedtuple

Mask = namedtuple("Mask", ["name", "kp", "des", "box", "box_points"])
DATA = namedtuple("DATA", ["name", "id", "address", "info", "box_size"])


def vid_handler(file):
    """
    Returns a FileVideoStream object using the given file.

    Args:
        file (str): path to the video file.

    Returns:
        FileVideoStream: FileVideoStream object.
    """
    return FileVideoStream(file, queue_size=128).start()


def webcam_handler(src):
    """
    Returns a WebcamVideoStream object using the given source.

    Args:
        src (int): source of the webcam.

    Returns:
        WebcamVideoStream: WebcamVideoStream object.
    """
    return WebcamVideoStream(src=src).start()


def save_descriptor_to_file(file, data):
    """
    Saves the given descriptor data to the given file.

    Args:
        file (str): path to the file.
        data (dict): data to be saved.

    Returns:
        None
    """
    pickle.dump(file, data)


def load_descriptor_from_file(file):
    """
    Loads the descriptor data from the given file.

    Args:
        file (str): path to the file.

    Returns:
        dict: descriptor data.
    """
    return pickle.load(file)


def load_img(filename: str, size: tuple = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads an image from the given file and returns it as a numpy array.

    Args:
        filename (str): path to the image file.
        size (tuple, optional): size of the image. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: image as a numpy array and the grayscale version of the image.
    """
    img = cv.imread(filename, cv.IMREAD_UNCHANGED)
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


def crop_img(img: np.ndarray, min_y: int, min_x: int, max_y: int, max_x: int) -> np.ndarray:
    """
    Crops the given image based on the given coordinates.

    Args:
        img (np.ndarray): image to be cropped.
        min_y (int): minimum y coordinate.
        min_x (int): minimum x coordinate.
        max_y (int): maximum y coordinate.
        max_x (int): maximum x coordinate.

    Returns:
        np.ndarray: cropped image.
    """
    return img[min_x:max_x, min_y:max_y, :]


def resize(img, width=None, height=None):
    """
    Resizes the given image to the given width and height.

    Args:
        img (np.ndarray): image to be resized.
        width (int, optional): width for the resized image. Defaults to None.
        height (int, optional): height for the resized image. Defaults to None.

    Returns:
        np.ndarray: resized image.
    """
    return imutils.resize(img, width=width, height=height)


def to_grayscale(img):
    """
    Converts the given image to grayscale.

    Args:
        img (np.ndarray): image to be converted.

    Returns:
        np.ndarray: grayscale image.
    """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def load_masks(path, compute_feature):
    """
    Loads all images from the given path, calculates keypoints
    and feature-descriptors and returns them as a dictionary of template objects.

    Args:
        path (str): path to the images.
        compute_feature (function): function to compute the feature-descriptors.

    Returns:
        dict: dictionary of template objects with their keypoints and feature-descriptors.
    """
    masks = {}
    for filename in glob.glob(path + "*.jpg"):
        img_mask, gray_mask = load_img(filename)
        kp_mask, des_mask = compute_feature(img_mask)
        h, w = gray_mask.shape
        name = Path(filename).stem[:-2]  # every mask is numbered _0-9 -> remove _%d
        new_mask = Mask(
            name,
            kp_mask,
            cv.UMat(des_mask),
            img_mask.shape[:2],
            np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            ),
        )
        if name in masks:
            masks[name].append(new_mask)
        else:
            masks[name] = [new_mask]
    return masks


def load_overlays(path, width=None, height=None):
    """
    Loads all images from the given path and returns them as a dictionary of overlay images.

    Args:
        path (str): path to the images.
        width (int, optional): width for the resized image. Defaults to None.
        height (int, optional): height for the resized image. Defaults to None.

    Returns:
        dict: dictionary of overlay images.
    """
    overlays = {}
    for filename in glob.glob(path + "*.png"):
        img, _ = load_img(filename)
        img = resize(img, width=width, height=height)
        name = Path(filename).stem
        overlays[name] = img
    return overlays


def read_yaml(filepath: str) -> dict:
    """Reads in a yaml file from disk based on the given filepath. This is only used to
    read in the config file

    Args:
        filepath (str): filepath to the yaml file

    Returns:
        dict: content of the yaml file as a dictionary
    """
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)
    return data
