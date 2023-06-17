from threading import Thread
import cv2 as cv
import numpy as np

import argparse
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import time
import datetime

import rendering as rd  # TODO: (TEMPORARY) Renderer should not be included in this file. In the end: All calls from Main


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


def resize_img(img, target_size):
    scale_percent = int(100 * target_size[0] / img.shape[0])
    scale_percent = max(scale_percent, int(100 * target_size[1] / img.shape[1]))

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)


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
    # cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))
    return cv.boundingRect(np.array(pts, dtype='int32').reshape((-1, 2)))


def convex_hull(pts: list[np.array((2, 1))]) -> np.array((-1, 1, 2)):
    # cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))
    return cv.convexHull(np.array(pts, dtype='int32').reshape((-1, 2)))  # .reshape((-1, 2))


# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
def match_flann(des, des2, kp, kp2, img, img2):
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

    if len(good) > 10:
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #dst = cv.perspectiveTransform(pts, M)
        #img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        #draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       #singlePointColor=None,
                       #matchesMask=matchesMask,  # draw only inliers
                       #flags=2)
        #img3 = cv.drawMatches(np.uint8(img), kp, np.uint8(img2), kp2, good, None, **draw_params)
        #plt.imshow(img3, 'gray'), plt.show()
        #return good[matchesMask]

    return good


def vid_handler(file):
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default=file,
                    help="path to input video file")
    args = vars(ap.parse_args())

    return FileVideoStream(args["video"], queue_size=128).start()


def webcam_handler():
    return WebcamVideoStream(src=0).start()


def main():

    compute_feature = compute_features_sift
    file = '../imgs/IMG_20230519_135110_1.jpg'

    # vid_stream(file)
    # return

    img2, gray2 = load_img(file, (600, 800))
    kp2, des2 = compute_feature(gray2)

    file = '../imgs/VID_20230612_151251.mp4'

    fvs = vid_handler(file)  # webcam_handler()  #
    count = 0
    fps = FPS().start()
    fps.update()

    # loop over frames from the video file stream
    while True:  # fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)

        frame = fvs.read()
        # if frame is None:
        #     break
        #
        frame = imutils.resize(frame, width=450)
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = np.dstack([frame, frame, frame])

        gray = np.float32(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        kp, des = compute_feature(gray)

        matches = match_flann(des, des2, kp, kp2, gray, img2)

        matches_pts = get_points_from_matched_keypoints(kp, matches)
        img_n = np.copy(frame)

        # img_n = rd.render_matches(img_n, kp, img2, kp2, matches)

        boundRect = bounding_box(matches_pts)
        cv.rectangle(img_n, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[2]), int(boundRect[3])), (0, 255, 0), 2)
        # img_n = rd.render_contours(img_n, [convex_hull(matches_pts)])


        if len(matches) > 10:  # 0.1 * float(len(kp2)):
            target = img_n
        else:
            target = frame


        # END rendering pipeline
        cv.putText(target, "approx. FPS: {:.2f}".format(fps._numFrames / ((datetime.datetime.now() - fps._start).total_seconds())), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # show the frame and update the FPS counter

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










    # while vid.isOpened():
    #
    #     ret, frame = vid.read()
    #
    #     if not frame is None:
    #         size = (600, 600)
    #         frame = resize_img(frame, size)
    #
    #         gray = np.float32(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    #         gray = cv.GaussianBlur(gray, (5, 5), 0)

    #         kp, des = compute_feature(gray)
    #
    #         matches = match_brute_force(des, des2)
    #
    #         matches_pts = get_points_from_matched_keypoints(kp, matches)
    #         img_n = np.copy(frame)
    #
    #         #img_n = rd.render_matches(img_n, kp, img2, kp2, matches)
    #
    #         boundRect = bounding_box(matches_pts)
    #         # cv.rectangle(img_n, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[2]), int(boundRect[3])), (0, 255, 0), 2)
    #         img_n = rd.render_contours(img_n, [convex_hull(matches_pts)])
    #
    #         if len(matches) > 60:  # 0.1 * float(len(kp2)):
    #             cv.imshow('frame', img_n)
    #         else:
    #             cv.imshow('frame', frame)
    #
    #         count = count + 1
    #     if cv.waitKey(10) & 0xFF == ord('q'):
    #         break
    #
    # vid.release()
    # cv.destroyAllWindows()

    return

    # compute_feature = compute_features_sift
    #
    # kp, des = compute_feature(img)
    # kp2, des2 = compute_feature(img2)
    #
    # # img3 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=4)  # DRAW_RICH_KEYPOINTS = 4
    # # cv.imshow('orb', img3)
    #
    # # img4 = cv.drawKeypoints(img2, kp, None, color=(0, 255, 0), flags=4)  # DRAW_RICH_KEYPOINTS = 4
    # # cv.imshow('orb2', img4)
    #
    # match_flann(des, des2, img, img2, kp, kp2)
    # # match_brute_force(des, des2, img, img2, kp, kp2)
    #
    # if cv.waitKey(0) & 0xff == 27:
    #     cv.destroyAllWindows()


main()
