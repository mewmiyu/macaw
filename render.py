import cv2 as cv
import numpy as np

# ORB: https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

# function for rendering the box
## see lecture 3d transformation slide 9 for help (rendering a bunny in a known space)
## -> we need to have a callibrated camera in that example, which is not possible in our case
## we also need either a mesh or a function, which gives us the metadata
def render_img(K, dist_coeff, vid):
    """
    Renders the square around the building
    :param K: camera callibration matrix
    :param dist_coeff:
    :param vid: a video of the building
    """
    while True:
        frame = vid.read()[1]
        ## here a function that is similar to the "detect markers function", which givs us e.g ids
        ids = None
        if ids is not None:
            ## we need to be able to compute the boundaries, then we can draw the square
            ## todo do we need edge detection rather than features? can we do both? our square needs the contours..
            ## https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            img = None
            contours = None
            cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        ## then we show the frame; theoretically, if we have a mesh we can use the function from the assignment


# function for tracking the features or getting the result from another method
def compute_features():
    # todo we need the camera callibration for the box with metadata, but contours for the square
    K = None
    dist_coeff = None

    return K, dist_coeff


# function for either rendering the box with metadata or getting the result from ogre
def render_metadata():
    return None


# main method
## note, if the ml algorithm doesnt work in the lecture "bildverarbeitung" on last slide is an algorithm for
## feature tracking
def main():
    K, dist_coeff = compute_features()
    vid = cv.VideoCapture('test.mp4')
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6x6_250)
    render_img(K, dist_coeff, vid)
