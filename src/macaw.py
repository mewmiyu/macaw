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
    mask = '../masks/mask_Hauptgebaeude_no_tree.jpg'  # TODO: Adapt for multiple masks
    input = '../imgs/VID_20230612_172955.mp4'  # 0  #
    compute_feature = features.compute_features_sift
    use_feature = 'SIFT'

    img_mask, gray_mask = utils.load_img(mask)
    kp_mask, des_mask = compute_feature(img_mask)
    h, w = gray_mask.shape
    masks = [features.Mask(kp_mask, des_mask, img_mask.shape[:2],  np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
)]

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
        count +=1
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        time_start = time.time()

        frame = fvs.read()
        if frame is None:
            break

        frame = utils.resize(frame, width=450)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray, (5, 5), 0)

        kp, des = compute_feature(frame_gray)

        boxes, labels, scores = detector.run_detector(frame)  # , showHits=True  # TODO: Do RGB channels nbeed to be swapped? rgb->bgr?
        hit, label, box = detector.filter_hits(boxes, labels, scores)
        # TODO: Filter + Crop boxes
        box_pixel = np.array(box * np.asarray(frame.shape[:2]), dtype=int).flatten()
        if hit and (box_pixel[2] - box_pixel[0]) * (box_pixel[3] - box_pixel[1]) > 0:  # hit and non-zero patch
            cropped = utils.crop_img(frame, *box_pixel.flatten())  # Test cropping and apply
            valid = False
            # tracking:
            if count % matching_rate != 0 and pts_f is not None:
                pts_f, pts_m, valid = features.track(last_frame_gray, frame_gray, pts_f, pts_m)

            # TODO: UMat
            # uframe = cv.UMat(cropped)  # cropped #

            if not valid:
                # rendering.display_image(cropped)
                matches, mask_id = features.match(des, masks)
                pts_f, pts_m = features.get_points_from_matched_keypoints(matches, kp, masks[mask_id].kp)

            last_frame_gray = frame_gray
            # homography
            dst = features.calc_bounding_box(matches, masks[mask_id], pts_f, pts_m)
            if dst is not None:
                frame = rendering.render_contours(frame, [np.int32(dst)])
                # TODO: Render meta data

        # show the frame and update the FPS counter
        rendering.render_text(frame, "FPS: {:.2f}".format(1.0 / (time.time() - time_start)))

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

