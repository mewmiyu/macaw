import argparse
import sys

import methods.object_detection as object_detection
import methods.labeling as labeling
import utils_macaw as utils
import features
import rendering
import video_player

import numpy as np
import cv2 as cv
import time

from methods.viewing import ImageViewer
from methods.eval import TorchImageProvider, PredictionsProvider
from utils.weights_loader import WeightsLoader


def macaw(input_file, path_masks, path_overlays, feature_type, model_checkpoint,
          device, root, annotations_path, num_classes,):
    """
    Main function of the MACAW project. This function is called from the main.py file.

    Args:
        input_file (str): Path to the input video file.
        path_masks (str): Path to the masks folder.
        path_overlays (str): Path to the overlays folder.
        feature_type (str): Type of features to be used. Either "ORB" or "SIFT".
        model_checkpoint (str): Path to the model checkpoint.
        device (str): Device to run the model on. Either "cpu" or "cuda".
        root (str): Path to the dataset.
        annotations_path (str): Path to the annotations.
        num_classes (int): Number of classes used during training.

    Returns:
        None
    """
    # TODO: Add parameters to the yaml file.
    detector_logging = True
    frame_width = 450
    matching_rate = 15

    if type(input_file) is int:
        fvs = utils.webcam_handler(input_file)  #
    else:
        fvs = utils.vid_handler(input_file)

    match feature_type:
        case "ORB":
            compute_feature = features.compute_features_orb
        case "SIFT":
            compute_feature = features.compute_features_sift
        case _:
            print("UNKNOWN_FEATURE_TYPE: Defaulting to SIFT features!")
            compute_feature = features.compute_features_sift

    # Load Masks
    masks = utils.load_masks(path_masks, compute_feature)
    frame_shape = fvs.read().shape

    # Load and rescale Overlays
    overlays = utils.load_overlays(
        path_overlays, width=int(0.75 * frame_shape[1])
    )  # width=int(0.75 * frame_width)
    overlay_shape = list(overlays.values())[0].shape
    if overlay_shape[0] > 0.5 * fvs.read().shape[0]:
        for i in overlays:
            overlays[i] = utils.resize(
                overlays[i], height=int(0.5 * fvs.read().shape[0])
            )

    # Initialize and start the VideoPlayer
    vid_out = video_player.VideoPlayerAsync(
        default_size=frame_shape[:2], target_fps=60
    ).start()

    # Initialize the detector
    model_predictor = PredictionsProvider(
        root, annotations_path, num_classes, model_checkpoint, device
    )

    # Ratio between full and computation resolution
    new_shape = utils.resize(fvs.read(), width=frame_width).shape
    ratio = frame_shape[0] / new_shape[0]
    overlay_pos = np.int32(
        (
            frame_shape[0] - overlay_shape[0] - 40,
            0.5 * frame_shape[1] - 0.5 * overlay_shape[1],
        )
    )

    # variables need across iterations
    last_frame_gray = None  # gray  # cv.UMat((1, 1))
    pts_f = None
    pts_m = None
    mask_id = None
    matches = None
    label = None
    count = -1

    # loop over frames from the video file stream
    while vid_out.running:
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

        render_target = cv.UMat(frame)
        frame = utils.resize(frame, width=frame_width)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.UMat(cv.GaussianBlur(frame_gray, (5, 5), 0))

        valid = False

        # tracking:
        if count != matching_rate and pts_f is not None and len(pts_f) > 0:
            pts_f, pts_m, valid = features.track(
                last_frame_gray, frame_gray, pts_f, pts_m, label
            )
        else:
            count = 0

        # Calculate the new bounding box after successful tracking
        if valid:
            bbox = features.calc_bounding_box(
                matches, masks[label][mask_id], pts_f, pts_m, label
            )

        last_frame_gray = frame_gray

        contours = []
        # Boxes are in the format XYXY
        if not valid or bbox is None:
            l = label
            pred = model_predictor(
                frame, silent=detector_logging
            )
            hit = pred[0]
            box_pixel = pred[1]
            label = pred[2]
            score = pred[3]

            if label is None:
                label = l
            if (
                    hit
                    and label in masks
                    and (box_pixel[2] - box_pixel[0]) * (box_pixel[3] - box_pixel[1]) > 0
            ):
                # hit and non-zero patch
                # Add the predicted bounding box from the model to the list for rendering
                contours.append((np.int32([[[box_pixel[0] * ratio, box_pixel[1] * ratio]],
                                [[box_pixel[2] * ratio, box_pixel[1] * ratio]],
                                [[box_pixel[2] * ratio, box_pixel[3] * ratio]],
                                [[box_pixel[0] * ratio, box_pixel[3] * ratio]], ]),
                        (255, 0, 0),))

                # Crop the img
                crop_offset = np.array([[[box_pixel[0], box_pixel[1]]]])
                cropped = utils.crop_img(frame, *box_pixel)  # Test cropping and apply
                cropped = cv.UMat(
                    cv.GaussianBlur(cv.cvtColor(cropped, cv.COLOR_BGR2GRAY), (5, 5), 0)
                )

                # match the features of the cropped img
                kp, des = compute_feature(cropped)
                matches, mask_id = features.match(des, masks[label], feature_type)
                pts_f, pts_m = features.get_points_from_matches(
                    matches, kp, masks[label][mask_id].kp
                )

                # get the bounding box (None, with/without homography -> depends on number of hits)
                bbox = features.calc_bounding_box(
                    matches, masks[label][mask_id], pts_f, pts_m, label
                )

        if bbox is not None:
            bbox += crop_offset  # bbox is calculated of the cropped image -> global coordinates by adding the offset
            contours.append((np.int32(bbox * ratio), (255, 0, 0)))
            # frame = rendering.render_fill_contours(render_target, np.int32(bbox))

        # Render all contours
        # for c in contours:
        #     render_target = rendering.render_contours(render_target, c[0], color=c[1])
        #     render_target = rendering.render_fill_contours(render_target, c[0], color=c[1])

        if len(contours) > 0:
            c = contours[-1]
            render_target = rendering.render_contours(render_target, c[0], color=c[1])
            render_target = rendering.render_fill_contours(
                render_target, c[0], color=c[1]
            )

        # Add Meta data:
        if len(contours) > 0:
            render_target = rendering.render_metadata(
                render_target, label, overlays, pos=overlay_pos, alpha=0.9
            )  # label

        # show the frame and update the FPS counter
        rendering.render_text(
            render_target,
            "FPS: {:.2f}".format(1.0 / (time.time() - time_start)),
            (10, frame_shape[0] - 10),
        )
        # cv.imshow("Frame", render_target)
        # cv.waitKey(1)

        vid_out.add(render_target)
        # Add Frame to the render Queue
        # while not vid_out.add(render_target) and vid_out.running:
        #     pass

    # do a bit of cleanup
    fvs.stop()
    vid_out.stop()
    cv.destroyAllWindows()
    sys.exit(0)


if __name__ == "__main__":
    """
    Main function of the macaw project.
    """
    parser = argparse.ArgumentParser("macaw")
    parser.add_argument("--config", help="The config file.")
    args = parser.parse_args()
    if "config" not in args:
        print("Failed to load config file.")
        exit(-1)
    cfg = utils.read_yaml(args.config)
    match cfg["METHOD"]["NAME"]:
        case "execute":
            if cfg["VIDEO"]["DOWNLOAD"]:
                weights_loader = WeightsLoader(cfg["VIDEO"]["MODEL_CHECKPOINT"])
                weights_loader()
            execute_cfg = dict(
                input_file=cfg["VIDEO"]["FILE_NAME"],
                path_masks=cfg["VIDEO"]["MASKS_PATH"],
                path_overlays=cfg["VIDEO"]["OVERLAYS_PATH"],
                feature_type=cfg["VIDEO"]["FEATURE_TYPE"],
                model_checkpoint=cfg["VIDEO"]["MODEL_CHECKPOINT"],
                device=cfg["VIDEO"]["DEVICE"],
                root=cfg["DATA"]["PATH"],
                annotations_path=cfg["DATA"]["ANNOTATIONS_PATH"],
                num_classes=cfg["TRAINING"]["NUM_CLASSES"],
            )
            macaw(**execute_cfg)
        case "train":
            object_detection.train(cfg)
        case "view":
            if cfg["EVALUATION"]["DOWNLOAD"]:
                weights_loader = WeightsLoader(cfg["EVALUATION"]["MODEL_CHECKPOINT"])
                weights_loader()
            eval_cfg = dict(
                root=cfg["DATA"]["PATH"],
                annotations=cfg["DATA"]["ANNOTATIONS_PATH"],
                num_classes=cfg["TRAINING"]["NUM_CLASSES"],
                model_checkpoint=cfg["EVALUATION"]["MODEL_CHECKPOINT"],
                device=cfg["EVALUATION"]["DEVICE"],
                batch_size=cfg["EVALUATION"]["BATCH_SIZE"],
                num_workers=cfg["EVALUATION"]["NUM_WORKERS"],
            )
            image_provider = TorchImageProvider(**eval_cfg)
            viewer = ImageViewer(image_provider)
            viewer()
        case "label":
            labeler = labeling.Labeler(cfg["DATA"]["ANNOTATIONS_PATH"])
            # We NEED to load all data, otherwise we won't have correct labels
            labeler(
                cfg["DATA"]["PATH"],
                cfg["DATA"]["SUPERCATEGORIES"],
                cfg["DATA"]["SUBCATEGORIES"],
                cfg["METHOD"]["MODE"],
            )
        case _:
            print(
                f"Unknown method: {cfg['METHOD']['NAME']}. Please use one of the"
                + "following: train, visualise, execute, label"
            )
            exit(-1)
