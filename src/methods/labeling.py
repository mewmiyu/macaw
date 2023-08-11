import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple
from methods.viewing import ImageViewer
from numpy.typing import NDArray
from utils.image_loader import DatasetImageProvider


class Labeler(ImageViewer):
    """This class allows to label images with a bounding box"""

    def __init__(self, annotations_path: str):
        """Initialises the Labeler class. The file in which the annotations are saved is
        given as an argument. All existing annotations are loaded, so the user can start
        with non-labeled images or review all labeled ones.

        Args:
            annotations_path (str): The path to the annotation file.

        Raises:
            ValueError: This error is raised if the format of the annotations file is
                wrong.
        """
        super().__init__(annotations_path)

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

    def __call__(
        self, data_path: str, folders: list[str], subfolders: list[str], mode: str
    ):
        """This function actually lets the user label the data. The path to the data is
        provided as an argument, as well as the classes und subclasses. Each image is
        then displayed in a matplotlib image on which the use can interact with. He can
        add points to the image, which correspond to the mask of the object. When
        pressing b, the points are converted into a bounding box. If the user presses n,
        the bounding box will be saved to the annotations and the next image will be
        displayed. When mode is set to review, the user can look at all the already
        labeled images. When pressing q, the application will quit.

        Args:
            data_path (str): The path to the dataset
            folders (list[str]): This list describes the main categories of the dataset
            subfolders (list[str]): This list describes the subcategories of the dataset
            mode (str): Either "annotate" or "review"
        """
        loader = DatasetImageProvider(folders, subfolders)
        images, labels, filenames, self.labeled_images["categories"] = loader(data_path)
        for i, image in enumerate(images):
            self.coords_x = []
            self.coords_y = []
            self.image = np.array(image)
            self.category_id = labels[i]
            self.title = filenames[i]

            ann_exists, img, ann = self.annotation_exists(self.title, self.category_id)
            self.image_id = img["id"] if img is not None else self.image_id
            if ann_exists and mode == "annotate":
                self.image_id += 1
                continue

            if not ann_exists and img is None:
                self.image_id = 1 + max(
                    [int(img_obj["id"]) for img_obj in self.labeled_images["images"]]
                )
                self.labeled_images["images"].append(
                    {
                        "id": self.image_id,
                        "file_name": self.title,
                        "height": self.image.shape[0],
                        "width": self.image.shape[1],
                    }
                )

            if ann_exists and mode == "review":
                bbox = ann["bbox"]
                target = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            else:
                target = None
            self.show_image(self.image, target, title=self.title)
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
            self.show_image(self.image, None, title=self.title)
        if event.key == "q":
            self.save_annotations(quit_on_success=True)
        if event.key == "b":
            minx = max(min(self.coords_x), 0)
            miny = max(min(self.coords_y), 0)
            maxx = min(max(self.coords_x), self.image.shape[1])
            maxy = min(max(self.coords_y), self.image.shape[0])
            width = maxx - minx
            height = maxy - miny

            self.show_image(self.image, (minx, miny, width, height), title=self.title)
            plt.scatter(self.coords_x, self.coords_y, zorder=2)

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

    def annotation_exists(
        self, image_name: str, category_id: int
    ) -> Tuple[bool, dict, int]:
        """This function checks if for a given image an annotation already exists. It
        also checks, if the annotation has the correct category. It returns a boolean,
        describing if the annotation exists, the image, if the image exists and the
        annotation, if it exists

        Args:
            image_name (str): The name of the image to check for
            category_id (int): The category the image should have

        Returns:
            Tuple[bool, dict, int]: A tuple, containing a boolean, whether the
            annotation exists, the image and the annotation itself
        """
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

    def save_annotations(self, quit_on_success: bool = False):
        """This function saves all current annotations. If 'quit_on_success' is true,
        the application will be exited after the save

        Args:
            quit_on_success (bool, optional): Whether to close the application after
                saving. Defaults to False.
        """
        try:
            with open(self.annotations_path, "w", encoding="utf-8") as f:
                json.dump(self.labeled_images, f)
            if quit_on_success:
                exit()
        except OSError:
            print("Annotation file could not be saved.")
