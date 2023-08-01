import json
import os
import torch.utils.data
import torchvision
from numpy.typing import ArrayLike
from PIL import Image
from torchvision import datapoints
from torchvision.ops.boxes import box_convert
from typing import Dict, Tuple

from torchvision.transforms.v2 import functional as F

torchvision.disable_beta_transforms_warning()
from torchvision.transforms.v2 import Compose


class CampusDataset(torch.utils.data.Dataset):
    """The CampusDataset inherits from torch.utils.data.Dataset and includes applying
    transformations to the loaded images. It also uses annotations in the COCO-format
    """

    def __init__(self, root: str = "", transforms: Compose = None):
        """Initialises the CampusDataset, which loads images from the root directory,
        using COCO-formatted annotations and applies transformations onto the images.

        Args:
            root (str, optional): Path to the dataset. Defaults to "".
            transforms (Compose, optional): Transformations to be applied on the images.
                Defaults to None.

        Raises:
            ValueError: Annotation file needs to be a json file with three keys:
                categories, images, annotations.
        """
        self.root = root
        self.transforms = transforms
        with open(root, mode="r") as f:
            annotations = json.load(f)
        keys = ["categories", "images", "annotations"]
        if any([key not in annotations for key in keys]):
            raise ValueError("Annotations file is in the wrong format.")

        self.categories, self.imgs, self.annotations = annotations.values()
        self.categories = {c["id"]: {**c, "id": c["id"] + 1} for c in self.categories}
        self.imgs = {i["id"]: i for i in self.imgs}

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, str | ArrayLike], Dict[str, torch.Tensor]]:
        """Returns the item in the dataset at a certain index

        Args:
            index (int): Index of the item in the dataset

        Returns:
            Tuple[Dict[str, str | ArrayLike], Dict[str, torch.Tensor]]: The image and
                its annotation in COCO format
        """
        id, img_id, ct_id, bbox, area, seg, iscrowd = self.annotations[index].values()

        category = self.categories[ct_id]

        boxes = torch.as_tensor([bbox], dtype=torch.float32)
        boxes = box_convert(boxes=boxes, in_fmt="xywh", out_fmt="xyxy")
        labels = torch.as_tensor([category["id"]], dtype=torch.int64)
        iscrowd = torch.as_tensor([iscrowd], dtype=torch.int64)
        area = torch.as_tensor([area], dtype=torch.int64)

        path_parts = category["name"].split("_")
        img_path = os.path.join("data", *path_parts, self.imgs[img_id]["file_name"])
        img = Image.open(img_path)

        target = {
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
            "boxes": datapoints.BoundingBox(
                boxes,
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=F.get_spatial_size(img),
            ),
            "labels": labels,
            "iscrowd": iscrowd,
            "area": area,
        }

        if self.transforms is not None:
            img, target["boxes"] = self.transforms(img, target["boxes"])

        return {"image": img, "filename": self.imgs[img_id]["file_name"]}, target

    def __len__(self):
        """Implements the __len__ method from the parent class

        Returns:
            int: Number of instances in the dataset
        """
        return len(self.imgs)
