import argparse
import sys

import methods.object_detection as object_detection
import methods.train as train
import methods.labeling as labeling
import utils_old
import feature_matching
import rendering

import numpy as np
import cv2 as cv
import time

from methods.viewing import ImageViewer
from methods.eval import TorchImageProvider, PredictionsProvider


def macaw():
    # Config:  # TODO: Outsource to base.yaml
    mask = "../masks/mask_Hauptgebaeude_no_tree.jpg"  # TODO: Adapt for multiple masks
    input = "../imgs/VID_20230612_172955.mp4"  # 0  #
    compute_feature = feature_matching.compute_features_sift

    img_mask, gray_mask = utils_old.load_img(mask)
    kp_mask, des_mask = compute_feature(img_mask)
    masks = [feature_matching.Mask(kp_mask, des_mask, img_mask.shape[:2])]

    if type(input) is int:
        fvs = utils_old.webcam_handler()  #
    else:
        fvs = utils_old.vid_handler(input)

    # loop over frames from the video file stream
    while True:  # fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        time_start = time.time()

        frame = fvs.read()
        if frame is None:
            break
        frame = utils_old.resize(frame, width=450)

        model_predictor = PredictionsProvider(
            "faster_rcnn-working-epoch.pt", "annotations_full.json"
        )
        _, _, prediction, _ = model_predictor(frame, silent=False)
        if len(prediction["boxes"]) > 0:
            box = np.array(prediction["boxes"][0].detach().to("cpu"))
            label = model_predictor.category_labels[prediction["labels"][0].item()]
            pass
        # hit, label, box = detector.filter_hits(
        #     prediction["boxes"], prediction["labels"], prediction["scores"]
        # )
        # TODO: Filter + Crop boxes
        if hit:  # hit
            box_pixel = np.array(box * np.asarray(frame.shape[:2]), dtype=int)
            cropped = utils_old.crop_img(
                frame, *box_pixel.flatten()
            )  # Test cropping and apply

            if np.all(np.array(cropped.shape) > 0):
                # rendering.display_image(cropped)
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
    cfg = utils_old.read_yaml(args.config)
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
