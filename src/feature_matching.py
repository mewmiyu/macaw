import cv2 as cv
import numpy as np

from imutils.video import FPS
import time

import utils
from collections import namedtuple

Mask = namedtuple("Mask", ["kp", "des", "box"])


def compute_features_sift(img: np.ndarray) -> tuple[cv.KeyPoint, np.ndarray]:  #  -> tuple[cv.KeyPoint]
    # SIFT: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

    sift = cv.SIFT_create()
    pic = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')  # cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
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
    # cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))


def convex_hull(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):
    # cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))
    return cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))


# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
def match_flann(des, des2, kp, kp2, mask_shape, MATCHING_THRESHOLD=20):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # we map from the template to the destination
    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if len(good) > 2 * MATCHING_THRESHOLD:  # TODO: Outsource homography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w = mask_shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, np.linalg.pinv(M))
        return dst
    if len(good) > MATCHING_THRESHOLD:
        return bounding_box(dst_pts)
    return None


def match(img: np.ndarray, masks: list[Mask], use_feature= 'SIFT'):
    match use_feature:
        case 'SIFT':
            compute_feature = compute_features_sift
        case 'ORB':
            compute_feature = compute_features_orb
        case _:
            compute_feature = compute_features_sift
    for mask in masks:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        kp, des = compute_feature(gray_img)

        dst = match_flann(des, mask.des, kp, mask.kp, mask.box, 20)
        return dst  # TODO: Support for list of masks -> return best match
    return img


if __name__ == "__main__":
    import rendering
    import detector

    mask = '../masks/mask_Hauptgebaeude_no_tree.jpg'
    input = '../imgs/VID_20230612_172955.mp4'

    compute_feature = compute_features_sift

    img_mask, gray_mask = utils.load_img(mask)
    kp_mask, des_mask = compute_feature(img_mask)
    masks = [Mask(kp_mask, des_mask, img_mask.shape[:2])]

    if type(input) is int:
        fvs = utils.webcam_handler()  #
    else:
        fvs = utils.vid_handler(input)

    fps = FPS().start()
    fps.update()

    # loop over frames from the video file stream
    while True:  # fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        time_start = time.time()

        frame = fvs.read()
        if frame is None:
            break
        frame = utils.resize(frame, width=450)

        target = frame.copy()

        boxes, labels, scores = detector.run_detector(frame)
        hit, box = detector.filter_hits(boxes, labels, scores)
        # TODO: Filter + Crop boxes
        if hit:
            dst = match(frame, masks)

            if dst is not None:  # 0.1 * float(len(kp_mask)):
                target = rendering.render_contours(target, [np.int32(dst)])

        # show the frame and update the FPS counter
        rendering.render_text(target, "approx. FPS: {:.2f}".format(1.0 / (time.time() - time_start)))

        # display the size of the queue on the frame
        cv.imshow('frame', target)
        cv.waitKey(1)
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv.destroyAllWindows()
    fvs.stop()

