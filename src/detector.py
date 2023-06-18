import time
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import rendering

import tempfile

from PIL import Image
from PIL import ImageOps

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"  # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
detector_ = hub.load(module_handle).signatures["default"]


def get_label_string(label, score):
    return "{}: {}%".format(label.decode("ascii"), int(100 * score))


def run_detector(img_in, detector=None, showHits = False):

    if detector is None:
        detector = detector_

    img = img_in
    if type(img) is np.ndarray:
        img = tf.convert_to_tensor(np.array(img)[:, :, 0:3])

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}
    # label_strings can be used for debugging
    # label_strings = [
    #     get_label_string(l, s)
    #     for l, s in zip(result["detection_class_entities"], result["detection_scores"])
    # ]
    # print(label_strings)

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)
    if showHits:
        image_with_boxes = rendering.draw_boxes(
            img.numpy(),
            result["detection_boxes"],
            result["detection_class_entities"],
            result["detection_scores"],
        )

        # Uncomment this to see boxes on image
        rendering.display_image(image_with_boxes)
    return (
        result["detection_boxes"],
        [l.decode("ascii") for l in result["detection_class_entities"]],
        result["detection_scores"],
    )


# returns if there is a hit and its bounding box (Min, Max) - pixel
def filter_hits(boxes, labels, scores) -> (bool, list[list[int, 2], list[int, 2]]):
    # TODO: Implement hit filtering
    return True, ((0, 0), (0, 0))


if __name__ == "__main__":
    def download_and_resize_image(url, new_width=256, new_height=256):
        _, filename = tempfile.mkstemp(suffix=".jpg")
        # response = urlopen(url)
        # image_data = response.read()
        # image_data = BytesIO(image_data)
        pil_image = Image.open(url)
        pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
        pil_image_rgb = pil_image.convert("RGB")
        pil_image_rgb.save(filename, format="JPEG", quality=90)
        print("Image downloaded to %s." % filename)

        return filename


    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img


    # By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
    image_url = ("https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg")
    downloaded_image_path = download_and_resize_image("../imgs/Untitled123.jpg", 1280, 856)
    boxes, labels, scores = run_detector(load_img(downloaded_image_path), showHits=True)
