import cv2 as cv
import numpy as np

# function for rendering the box
## see lecture 3d transformation slide 9 for help (rendering a bunny in a known space)
## -> we need to have a callibrated camera in that example, which is not possible in our case
## we also need either a mesh or a function, which gives us the metadata
def render_img1(K, dist_coeff, vid):
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


def render_img(img = None, ):
    if img is None:
        size = (600, 800)
        img = np.zeros(size)

    # draw all contours

    # draw all meta-data

    return img


def render_contours(img: np.ndarray, contours) -> np.ndarray:
    return cv.drawContours(img, contours, 0, (0, 255, 0), 2)


def render_matches(img, kp, img2, kp2, matches):
    return cv.drawMatches(img, kp, img2, kp2, matches, None)


# function for either rendering the box with metadata or getting the result from ogre
def render_metadata(img: np.ndarray) -> np.ndarray:
    return img


# main method
## note, if the ml algorithm doesnt work in the lecture "bildverarbeitung" on last slide is an algorithm for
## feature tracking
# def main():
#     K, dist_coeff = compute_features()
#     vid = cv.VideoCapture('test.mp4')
#     dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6x6_250)
#     render_img(K, dist_coeff, vid)
