import cv2 as cv
import numpy as np

# from datetime import datetime
#
# start_time = datetime.now()
#
# # INSERT YOUR CODE
#
# time_elapsed = datetime.now() - start_time
#
# print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


def load_img(filename: str, size: tuple = None) -> tuple[np.ndarray, np.ndarray]:
    img = cv.imread(filename)
    if size:
        scale_percent = int(100 * size[0] / img.shape[0])
        scale_percent = max(scale_percent, int(100 * size[1] / img.shape[1]))

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return img, gray


def compute_SIFT(img: np.ndarray):  #  -> tuple[cv.KeyPoint]
    # SIFT: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

    sift = cv.SIFT_create()
    pic = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')  # cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(pic, None)

    return kp, des


def compute_harris(img: np.ndarray, threshold=0.01):
    # Harris : https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html

    dst = cv.cornerHarris(img, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img2 = img.copy()
    img2[dst > threshold * dst.max()] = 0
    return np.argwhere(dst > threshold * dst.max()), img2


def compute_ORB(img: np.ndarray):
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


def convex_hull(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):

    return cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))


# https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
def match_brute_force(des, des2, img, img2, kp, kp2):
    # brute force matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # draw first 50 matches
    match_img = cv.drawMatches(img, kp, img2, kp2, matches[:50], None)
    cv.imshow('Matches', match_img)


# https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
def match_flann(des, des2, img, img2, kp, kp2):
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des, des2, k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            # p1 = kp[m.queryIdx].pt
            # matches_points.append(np.array(p1))

    matches_pts = get_points_from_matched_keypoints(kp, good_matches)
    hull = convex_hull(matches_pts)

    match_img = cv.drawMatches(img, kp, img2, kp2, good_matches, None)

    match_img = cv.drawContours(match_img, [hull], 0, (0, 255, 0), 2)

    cv.imshow('Matches', match_img)

    if len(good_matches) > 1:
        src_points = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
        corrected_img = cv.warpPerspective(img, m, (img2.shape[1], img2.shape[0]))
        # cv.imshow('Corrected image', corrected_img)


def main():
    file = '../imgs/20230522_114217683_iOS.jpg'
    img, gray = load_img(file, (600, 800))

    file = '../imgs/IMG_20230519_135110_1.jpg'
    img2, gray2 = load_img(file, (600, 800))

    compute_feature = compute_SIFT

    kp, des = compute_feature(img)
    kp2, des2 = compute_feature(img2)

    # img3 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=4)  # DRAW_RICH_KEYPOINTS = 4
    # cv.imshow('orb', img3)

    # img4 = cv.drawKeypoints(img2, kp, None, color=(0, 255, 0), flags=4)  # DRAW_RICH_KEYPOINTS = 4
    # cv.imshow('orb2', img4)

    match_flann(des, des2, img, img2, kp, kp2)
    # match_brute_force(des, des2, img, img2, kp, kp2)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


main()