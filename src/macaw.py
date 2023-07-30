import sys

# import methods.train as train
import utils
import features
import detector
import rendering

import numpy as np
import cv2 as cv
import time


def macaw():

    # Config:  # TODO: Outsource to base.yaml
    path_masks = '../masks/'
    path_overlays = '../masks/overlay/'
    input = '../imgs/VID_20230612_172955.mp4'  # 0  #

    use_feature = 'SIFT'
    frame_width = 450

    masks = utils.load_masks(path_masks)
    overlays = utils.load_overlays(path_overlays, int(0.75 * frame_width))

    if type(input) is int:
        fvs = utils.webcam_handler()  #
    else:
        fvs = utils.vid_handler(input)

        match use_feature:
            case 'ORB':
                compute_feature = features.compute_features_orb
            case 'SIFT':
                compute_feature = features.compute_features_sift
            case _:
                compute_feature = features.compute_features_sift

    last_frame_gray = None  # gray  # cv.UMat((1, 1))
    pts_f = None
    pts_m = None
    mask_id = None
    matches = None

    count = -1

    matching_rate = 10
    # loop over frames from the video file stream
    while True:  # fvs.more():
        count += 1
        bbox = None
        crop_offset = np.array([[[0, 0]]])

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        time_start = time.time()

        frame = fvs.read()
        if frame is None:
            break

        frame = utils.resize(frame, width=frame_width)
        frame_size = frame.shape
        frame_umat = cv.UMat(frame)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.UMat(cv.GaussianBlur(frame_gray, (5, 5), 0))

        valid = False
        # tracking:
        if count % matching_rate != 0 and pts_f is not None and len(pts_f) >0:
            pts_f, pts_m, valid = features.track(last_frame_gray, frame_gray, pts_f, pts_m)
            bbox = features.calc_bounding_box(matches, masks[mask_id], pts_f, pts_m)

        last_frame_gray = frame_gray

        boxes, labels, scores = detector.run_detector(frame)  # , showHits=True
        hit, label, box = detector.filter_hits(boxes, labels, scores)
        # TODO: Filter + Crop boxes
        # TODO: Check order of the new box! (xy min max)
        box_pixel = np.array(box * np.asarray(frame.shape[:2]), dtype=int).flatten()
        if not valid and hit and (box_pixel[2] - box_pixel[0]) * (box_pixel[3] - box_pixel[1]) > 0:  # hit and non-zero patch

            # Crop the img
            crop_offset = np.array([[[box_pixel[1], box_pixel[0]]]])
            cropped = utils.crop_img(frame, *box_pixel.flatten())  # Test cropping and apply
            frame_gray = cv.UMat(cv.GaussianBlur(cv.cvtColor(cropped, cv.COLOR_BGR2GRAY), (5, 5), 0))

            # match the features of the cropped img
            kp, des = compute_feature(frame_gray)
            matches, mask_id = features.match(des, masks)
            pts_f, pts_m = features.get_points_from_matched_keypoints(matches, kp, masks[mask_id].kp)

            # gt the bounding box (None, with/without homography -> depends on number of hits)
            bbox = features.calc_bounding_box(matches, masks[mask_id], pts_f, pts_m)

        if bbox is not None:
            bbox += crop_offset  # bbox is calculated of the croped image -> global coordinates by adding the offset
            frame = rendering.render_contours(frame_umat, np.int32(bbox))
            frame = rendering.render_fill_contours(frame, np.int32(bbox))
            frame = rendering.render_metadata(frame, 'mask_Hauptgebaeude_no_tree', overlays['hauptgebaeude_overlay'])  # label

        # show the frame and update the FPS counter
        rendering.render_text(frame, "FPS: {:.2f}".format(1.0 / (time.time() - time_start)), (10, frame_size[0] - 10))

        # display the size of the queue on the frame
        cv.imshow('frame', frame)
        cv.waitKey(1)

    # do a bit of cleanup
    cv.destroyAllWindows()
    fvs.stop()
    return


if __name__ == '__main__':
    macaw()

    if (len(sys.argv)) != 2:
        print("Failed to load config file.")
        exit(-1)
    cnfg = utils.read_yaml(sys.argv[1])
    match cnfg['METHOD']:
        # case 'train':
        #     train.train(cnfg)
        case 'execute':
            macaw()
        case _:
            print(f"Unknown method: {cnfg['METHOD']}. Please use one of the following: train")
            exit(-1)

