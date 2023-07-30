import cv2 as cv
import numpy as np

from imutils.video import FPS
import time

import utils

TRACKING_THRESHOLD = 20
MATCHING_THRESHOLD = 20
MATCH_DISTANCE = 0.7


def compute_features_sift(img: np.ndarray) -> tuple[cv.KeyPoint, np.ndarray]:  # -> tuple[cv.KeyPoint]
    # SIFT: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

    sift = cv.SIFT_create()
    pic = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)  # .astype('uint8')  # cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
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


def get_points_from_matched_keypoints(matches_accepted, kp, kp2):
    pts1 = np.float32([kp[m.queryIdx].pt for m in matches_accepted]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches_accepted]).reshape(-1, 1, 2)
    return pts1, pts2


def bounding_box(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):
    br = cv.boundingRect(np.array(pts, dtype='int32').reshape((-1, 2)))
    return np.array([[[br[0], br[1]]], [[br[2], br[1]]], [[br[2], br[3]]], [[br[0], br[3]]]])


def convex_hull(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):
    return cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))


# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
def match_flann(des, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    matches_accepted = []
    for m, n in matches:
        if m.distance < MATCH_DISTANCE * n.distance:
            matches_accepted.append(m)
    return matches_accepted


def estimate_homography(pts_src, points_st):
    m, mask = cv.findHomography(pts_src, points_st, cv.RANSAC, 5.0)  # returns M, mask
    return m, mask


def match(des, masks, target='mask_Hauptgebaeude_no_tree', use_feature='SIFT'):
    matches_best = None
    matches_best_nr = -1
    mask_id = target  # TODO: Remove
    for idx, mask in enumerate(masks):
        mask = masks[target]  # TODO: Remove
        matches_accepted = match_flann(des, mask.des)

        if len(matches_accepted) > matches_best_nr:
            matches_best = matches_accepted
            matches_best_nr = len(matches_accepted)
            mask_id = idx
            mask_id = target  # TODO: Remove
        # return src_pts
        break  # TODO: Remove
    return matches_best, mask_id  # TODO: Support for list of masks -> return best match


def calc_bounding_box(matches_accepted, mask, src_pts, mask_pts):
    # With enough matches: Estimate Homography
    if len(matches_accepted) > 2 * MATCHING_THRESHOLD:
        m, msk = estimate_homography(src_pts, mask_pts)
        dst = cv.perspectiveTransform(mask.box_points, np.linalg.pinv(m))
        return dst
    # With slightly fewer hits: Fit bounding box
    if len(matches_accepted) > int( 0.75 * MATCHING_THRESHOLD):
        return bounding_box(mask_pts)
    # Else: No matches
    return None


def track(img_old, img_new, pts_old, pts_mask_old):
    pts_new, st, err = cv.calcOpticalFlowPyrLK(img_old, img_new, pts_old, None, minEigThreshold=0.1)
    good_new = None
    mask_new = None
    # good_old = []
    # img = np.zeros_like(img_old)
    valid = False  # Check if enough points are tracked
    if pts_new is not None:
        good_new = pts_new.get()[st.get()[:, 0] == 1]
        mask_new = pts_mask_old[st.get()[:, 0] == 1]
        # good_old = pts_old[st == 1]
        #
        # mask = np.zeros_like(img_old)
        # color = np.random.randint(0, 255, (100, 3))

        # draw the tracks
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #     frame = cv.circle(img_old, (int(a), int(b)), 5, color[i].tolist(), -1)
        #     img = cv.add(frame, mask)
        #
        # cv.imshow('frame', img)
        # cv.waitKey(1)
    if len(good_new) >= TRACKING_THRESHOLD:
        valid = True

    return good_new, mask_new, valid
