import argparse
import sys

import methods.train as train
import methods.labeling as labeling
import utils
import feature_matching
import detector
import rendering

import numpy as np
import cv2 as cv
import time


def macaw():
    # Config:  # TODO: Outsource to base.yaml
    mask = "../masks/mask_Hauptgebaeude_no_tree.jpg"  # TODO: Adapt for multiple masks
    input = "../imgs/VID_20230612_172955.mp4"  # 0  #
    compute_feature = feature_matching.compute_features_sift

    img_mask, gray_mask = utils.load_img(mask)
    kp_mask, des_mask = compute_feature(img_mask)
    masks = [feature_matching.Mask(kp_mask, des_mask, img_mask.shape[:2])]

    if type(input) is int:
        fvs = utils.webcam_handler()  #
    else:
        fvs = utils.vid_handler(input)

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

        boxes, labels, scores = detector.run_detector(
            frame
        )  # , showHits=True  # TODO: Do RGB channels nbeed to be swapped? rgb->bgr?
        hit, label, box = detector.filter_hits(boxes, labels, scores)
        # TODO: Filter + Crop boxes
        if hit:
            box_pixel = np.array(box * np.asarray(frame.shape[:2]), dtype=int)
            cropped = utils.crop_img(
                frame, *box_pixel.flatten()
            )  # Test cropping and apply

            if np.all(np.array(cropped.shape) > 0):
                rendering.display_image(cropped)
                dst = feature_matching.match(cropped, masks)

                if dst is not None:
                    frame = rendering.render_contours(frame, [np.int32(dst)])
                    # TODO: Render meta data

        # show the frame and update the FPS counter
        rendering.render_text(
            frame, "FPS: {:.2f}".format(1.0 / (time.time() - time_start))
        )

        # display the size of the queue on the frame
        cv.imshow("frame", frame)
        cv.waitKey(1)

    # do a bit of cleanup
    cv.destroyAllWindows()
    fvs.stop()
    return


if __name__ == "__main__":
    # macaw()

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("--config", help="The config file.")
    args = parser.parse_args()
    if "config" not in args:
        print("Failed to load config file.")
        exit(-1)
    cnfg = utils.read_yaml(args.config)
    match cnfg["METHOD"]["NAME"]:
        case "train":
            train.train(cnfg)
        case "execute":
            macaw()
        case "labeling":
            labeler = labeling.Labeler("annotations.json")
            # We NEED to load all data, otherwise we won't have correct labels
            labeler("data", cnfg["METHOD"]["MODE"])
        case _:
            print(
                f"Unknown method: {cnfg['METHOD']}. Please use one of the following: train"
            )
            exit(-1)
