import argparse
import sys

import methods.object_detection as object_detection
import methods.train as train
import methods.labeling as labeling
import utils_macaw as utils
import features
import rendering

import numpy as np
import cv2 as cv
import time

from methods.viewing import ImageViewer
from methods.eval import TorchImageProvider, PredictionsProvider


def macaw():
    # Config:  # TODO: Outsource to base.yaml
    path_masks = '../masks/'
    path_overlays = '../masks/overlay/'
    input = '../imgs/VID_20230612_172955.mp4'  # 0  #

    use_feature = 'SIFT'

    masks = utils.load_masks(path_masks)
    overlays = utils.load_overlays(path_overlays)

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
    model_predictor = PredictionsProvider(
        "faster_rcnn-working-epoch.pt", "annotations_full.json"
    )
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

        frame = utils.resize(frame, width=450)
        frame_size = frame.shape
        frame_umat = cv.UMat(frame)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.UMat(cv.GaussianBlur(frame_gray, (5, 5), 0))

        valid = False
        # tracking:
        if count % matching_rate != 0 and pts_f is not None and len(pts_f) > 0:
            pts_f, pts_m, valid = features.track(last_frame_gray, frame_gray, pts_f, pts_m)
            bbox = features.calc_bounding_box(matches, masks[mask_id], pts_f, pts_m)

        last_frame_gray = frame_gray

        #boxes, labels, scores = detector.run_detector(frame)  # , showHits=True
        #hit, label, box = detector.filter_hits(boxes, labels, scores)
        _, _, prediction, _ = model_predictor(frame, silent=False)
        if len(prediction["boxes"]) > 0:
            box = np.array(prediction["boxes"][0].detach().to("cpu"))
            label = model_predictor.category_labels[prediction["labels"][0].item()]
            pass

        # TODO: Filter + Crop boxes
        # TODO: Check order of the new box! (xy min max)
        box_pixel = np.array(box * np.asarray(frame.shape[:2]), dtype=int).flatten()
        if not valid and hit and (box_pixel[2] - box_pixel[0]) * (
                box_pixel[3] - box_pixel[1]) > 0:  # hit and non-zero patch

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
            frame = rendering.render_metadata(frame, 'mask_Hauptgebaeude_no_tree')  # label

        # show the frame and update the FPS counter
        rendering.render_text(frame, "FPS: {:.2f}".format(1.0 / (time.time() - time_start)), (10, frame_size[0] - 10))

        # display the size of the queue on the frame
        cv.imshow('frame', frame)
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
    cfg = utils.read_yaml(args.config)
    match cfg["METHOD"]["NAME"]:
        case "execute":
            macaw()
        case "train":
            object_detection.train(cfg)
        case "view":
            eval_cfg = dict(
                annotations="annotations_full.json",
                model_checkpoint=cfg["EVALUATION"]["CHECKPOINT"],
                device=cfg["EVALUATION"]["DEVICE"],
                batch_size=cfg["EVALUATION"]["BATCH_SIZE"],
                num_workers=cfg["EVALUATION"]["NUM_WORKERS"],
            )
            image_provider = TorchImageProvider(**eval_cfg)
            viewer = ImageViewer(image_provider)
            viewer()
        case "label":
            labeler = labeling.Labeler("annotations_full.json")
            # We NEED to load all data, otherwise we won't have correct labels
            labeler(
                cfg["DATA"]["PATH"],
                cfg["DATA"]["SUPERCATEGORIES"],
                cfg["DATA"]["SUBCATEGORIES"],
                cfg["METHOD"]["MODE"],
            )
        case _:
            print(
                f"Unknown method: {cfg['METHOD']['NAME']}. Please use one of the following: train, visualise, execute, label"
            )
            exit(-1)
