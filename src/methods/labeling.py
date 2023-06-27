import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import utils


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
            if any(
                [
                    key not in self.labeled_images
                    for key in ["categories", "images", "annotations"]
                ]
            ):
                raise ValueError("Annotations file is in the wrong format.")
        else:
            self.labeled_images = {
                "categories": self.categories,
                "images": [],
                "annotations": [
                    {"id": self.annotation_id, "image_id": 0, "category_id": -1}
                ],
            }
        self.image_id = len(self.labeled_images["images"])
        self.coords_x = []
        self.coords_y = []

    def __call__(self, data_folder):
        images, labels, filenames, self.labeled_images["categories"] = utils.load_data(
            data_folder
        )
        for i, image in enumerate(images):
            self.coords_x = []
            self.coords_y = []
            self.image = np.array(image)
            self.category_id = labels[i]
            self.title = filenames[i]

            if self.image_exists(self.title, self.category_id):
                continue

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
            plt.show(block=True)
            self.image_id += 1

    def on_press(self, event):
        print("press", event.key)
        if event.key == "x":
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
            except OSError:
                print("Annotation file could not be saved.")
            finally:
                exit()
        if event.key == "b":
            minx = np.maximum(np.min(self.coords_x), 0)
            miny = np.maximum(np.min(self.coords_y), 0)
            maxx = np.minimum(np.max(self.coords_x), self.image.shape[0])
            maxy = np.minimum(np.max(self.coords_y), self.image.shape[1])
            width = maxy - miny
            height = maxx - minx

            plt.clf()
            plt.scatter(self.coords_x, self.coords_y, zorder=2)
            plt.imshow(self.image, zorder=1)
            plt.plot([minx, minx], [miny, maxy], "red", zorder=2)
            plt.plot([maxx, maxx], [miny, maxy], "red", zorder=2)
            plt.plot([minx, maxx], [miny, miny], "red", zorder=2)
            plt.plot([minx, maxx], [maxy, maxy], "red", zorder=2)
            plt.title(self.title)
            plt.draw()

            last_annotation = self.labeled_images["annotations"][-1]
            self.annotation_id = last_annotation["id"] + 1
            new_annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": self.category_id,
                "bbox": [minx, miny, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0,
            }
            if self.image_id != last_annotation["image_id"]:
                self.labeled_images["annotations"].append(new_annotation)
            else:
                self.labeled_images["annotations"][-1] = new_annotation

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

    def image_exists(self, file_name, category_id) -> bool:
        # Check if an image with the same file_name exists in images
        for image in self.labeled_images["images"]:
            if image["file_name"] == file_name:
                # Check if there is a corresponding annotation with the same category_id
                for annotation in self.labeled_images["annotations"]:
                    if (
                        annotation["image_id"] == image["id"]
                        and annotation["category_id"] == category_id
                    ):
                        return True  # Image with same file_name and category_id exists

        return False  # Image with same file_name and category_id does not exist
