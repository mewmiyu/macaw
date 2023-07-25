import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import utils

from typing import Tuple
from utils_package.image_loader import ImageLoader


class Labeler:
    def __init__(self, annotations_path):
        self.image = np.zeros((300, 300), dtype=np.float32)
        self.categories = []
        self.category_id = -1
        self.annotation_id = -1
        self.title = ""
        self.annotations_path = annotations_path
        if os.path.isfile(self.annotations_path):
            with open(self.annotations_path, "r", encoding="utf-8") as f:
                self.labeled_images = json.load(f)
            keys = ["categories", "images", "annotations"]
            if any([key not in self.labeled_images for key in keys]):
                raise ValueError("Annotations file is in the wrong format.")
        else:
            self.labeled_images = {
                "categories": self.categories,
                "images": [],
                "annotations": [],
            }
        self.image_id = len(self.labeled_images["images"])
        self.coords_x = []
        self.coords_y = []

    def __call__(self, data_folder: str, mode: str):
        loader = ImageLoader()
        images, labels, filenames, self.labeled_images["categories"] = loader.load_data(
            data_folder
        )
        for i, image in enumerate(images):
            self.coords_x = []
            self.coords_y = []
            self.image = np.array(image.permute((1, 2, 0)))
            self.category_id = labels[i]
            self.title = filenames[i]

            ann_exists, img, ann = self.annotation_exists(self.title, self.category_id)
            self.image_id = img["id"] if img is not None else self.image_id
            if ann_exists and mode == "annotate":
                self.image_id += 1
                continue

            if not ann_exists and img is None:
                self.image_id = max([int(img_obj["id"]) for img_obj in self.labeled_images["images"]])+1
                self.labeled_images["images"].append(
                    {
                        "id": self.image_id,
                        "file_name": self.title,
                        "height": self.image.shape[0],
                        "width": self.image.shape[1],
                    }
                )

            fig = plt.figure(figsize=(10, 10))
            fig.canvas.mpl_connect("button_press_event", self.onclick)
            fig.canvas.mpl_connect("key_press_event", self.on_press)
            plt.title(self.title)
            plt.imshow(self.image, zorder=1)
            if ann_exists and mode == "review":
                bbox = ann["bbox"]
                self.draw_bbox(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            plt.show(block=True)
            self.image_id += 1

    def on_press(self, event):
        print("press", event.key)
        if event.key == "n":
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(self.labeled_images, f)
            print(self.coords_x, self.coords_y)
            plt.close()
        if event.key == "r":
            self.coords_x = []
            self.coords_y = []
            plt.clf()
            plt.imshow(self.image, zorder=1)
            plt.title(self.title)
            plt.draw()
        if event.key == "q":
            try:
                with open(self.annotations_path, "w", encoding="utf-8") as f:
                    json.dump(self.labeled_images, f)
                exit()
            except OSError:
                print("Annotation file could not be saved.")
        if event.key == "b":
            minx = max(min(self.coords_x), 0)
            miny = max(min(self.coords_y), 0)
            maxx = min(max(self.coords_x), self.image.shape[1])
            maxy = min(max(self.coords_y), self.image.shape[0])
            width = maxx - minx
            height = maxy - miny

            plt.clf()
            plt.scatter(self.coords_x, self.coords_y, zorder=2)
            plt.title(self.title)
            plt.imshow(self.image, zorder=1)
            self.draw_bbox(minx, miny, maxx, maxy)

            self.annotation_id = 1 + max(
                [*[ann["id"] for ann in self.labeled_images["annotations"]], -1]
            )

            new_annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": self.category_id,
                "bbox": [minx, miny, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0,
            }

            annotations_for_image = [
                ann["image_id"] == self.image_id
                for ann in self.labeled_images["annotations"]
            ]
            if not any(annotations_for_image):
                self.labeled_images["annotations"].append(new_annotation)
            else:
                i_annotation = annotations_for_image.index(1)
                self.labeled_images["annotations"][i_annotation] = new_annotation

    def onclick(self, event):
        i_x, i_y = event.xdata, event.ydata
        print(f"x = {i_x}, y = {i_y}")
        if i_x is not None and i_y is not None:
            self.coords_x.append(i_x)
            self.coords_y.append(i_y)
            plt.clf()
            plt.scatter(self.coords_x, self.coords_y, zorder=2)
            plt.imshow(self.image, zorder=1)
            plt.title(self.title)
            plt.draw()

    def annotation_exists(self, image_name, category_id) -> Tuple[bool, dict, dict]:
        # Check if an image with the same file_name exists in images
        for image in self.labeled_images["images"]:
            if image["file_name"] == image_name:
                # Check if there is a corresponding annotation with the same category_id
                for annotation in self.labeled_images["annotations"]:
                    if (
                        annotation["image_id"] == image["id"]
                        and annotation["category_id"] == category_id
                    ):
                        # Image with same file_name and category_id exists
                        return True, image, annotation
                return False, image, None

        return False, None, None  # Annotation does not exist

    def draw_bbox(self, minx, miny, maxx, maxy):
        plt.plot([minx, minx], [miny, maxy], "red", zorder=2)
        plt.plot([maxx, maxx], [miny, maxy], "red", zorder=2)
        plt.plot([minx, maxx], [miny, miny], "red", zorder=2)
        plt.plot([minx, maxx], [maxy, maxy], "red", zorder=2)
        plt.draw()
