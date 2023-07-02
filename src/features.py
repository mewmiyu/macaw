import cv2 as cv
import numpy as np

from imutils.video import FPS
import time

import utils
from collections import namedtuple

Mask = namedtuple("Mask", ["kp", "des", "box", "box_points"])


def compute_features_sift(img: np.ndarray) -> tuple[cv.KeyPoint, np.ndarray]:  #  -> tuple[cv.KeyPoint]
    # SIFT: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

    sift = cv.SIFT_create()
    pic = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) #.astype('uint8')  # cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(pic, None)

    return kp, des


def compute_features_harris(img: np.ndarray, threshold=0.01):
    # Harris : https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html

    dst = cv.cornerHarris(img, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img2 = img.copy()
    img2[dst > threshold * dst.max()] = 0
    return np.argwhere(dst > threshold * dst.max()), img2


def compute_features_orb(img: np.ndarray) -> tuple[cv.KeyPoint, np.ndarray]:
    # ORB: https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    return kp, des


def get_points_from_matched_keypoints(kp, matches):
    pts = []
    for m in matches:
        p1 = kp[m.queryIdx].pt
        pts.append(np.array(p1))
    return pts


def bounding_box(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):
    br = cv.boundingRect(np.array(pts, dtype='int32').reshape((-1, 2)))
    return np.array([[[br[0], br[1]]], [[br[2], br[1]]],[[br[2], br[3]]], [[br[0], br[3]]]])

def convex_hull(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):
    return cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))


# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
def match_flann(des, des2, kp, kp2, mask_shape, MATCHING_THRESHOLD=20):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    matches_accepted = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            matches_accepted.append(m)
    return matches_accepted


def estimate_homography(pts_src, points_st):
    m, mask = cv.findHomography(pts_src, points_st, cv.RANSAC, 5.0)  # returns M, mask
    return m, mask


def match(img, masks: list[Mask], use_feature='SIFT', MATCHING_THRESHOLD=10):
    match use_feature:
        case 'ORB':
            compute_feature = compute_features_orb
        case 'SIFT':
            compute_feature = compute_features_sift
        case _:
            compute_feature = compute_features_sift

    for mask in masks:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        kp, des = compute_feature(gray_img)

        matches_accepted = match_flann(des, mask.des, kp, mask.kp, mask.box)
        # return src_pts

        # we map from the template to the destination
        src_pts = np.float32([kp[m.queryIdx].pt for m in matches_accepted]).reshape(-1, 1, 2)
        mask_pts = np.float32([mask.kp[m.trainIdx].pt for m in matches_accepted]).reshape(-1, 1, 2)


        #track(img, img, src_pts)

        # With enough matches: Estimate Homography
        if len(matches_accepted) > 2 * MATCHING_THRESHOLD:
            m, msk = estimate_homography(src_pts, mask_pts)
            dst = cv.perspectiveTransform(mask.box_points, np.linalg.pinv(m))
            return dst
        # With slightly fewer hits: Fit bounding box
        if len(matches_accepted) > MATCHING_THRESHOLD:
            return bounding_box(mask_pts)
        # Else: No matches
        return None  # TODO: Support for list of masks -> return best match


def track(img_old, img_new, pts_old):
    pts_new = []
    pts_new, st, err = cv.calcOpticalFlowPyrLK(img_old, img_new, pts_old, None, minEigThreshold=0.1)
    return 1

