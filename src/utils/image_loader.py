import os
from typing import Any

from PIL import Image
from src.utils.preprocess import get_transform


class ImageProvider:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()


class DatasetImageProvider(ImageProvider):
    def __init__(self, folders, subfolders) -> None:
        self.supercategories = folders
        self.subcategories = subfolders

    def __call__(self, data_path):
        """
        This function loads all images from the data directory.
        Each new directory creates a new label, so images from
        the same category should be in the same directory
        """
        label = 0
        categories = []
        labels = []
        images = []
        file_names = []
        for supercategory in self.supercategories:
            supercategory_path = os.path.join(data_path, supercategory)
            subdirs = [
                dir
                for dir in os.listdir(supercategory_path)
                if os.path.isdir(os.path.join(supercategory_path, dir))
            ]

            for subcategory in self.subcategories:
                if subcategory not in subdirs:
                    continue

                subcategory_path = os.path.join(supercategory_path, subcategory)
                files = [
                    file
                    for file in os.listdir(subcategory_path)
                    if os.path.isfile(os.path.join(subcategory_path, file))
                    and file.lower() != ".ds_store"
                ]
                if len(files) == 0:
                    continue

                for file in files:
                    image = Image.open(os.path.join(subcategory_path, file))
                    image_tensor = get_transform(train=False)(image).to("cpu")

                    labels.append(label)
                    images.append(image_tensor)
                    file_names.append(file)

                label += 1
                category = "_".join([supercategory, subcategory])
                categories.append(
                    {
                        "id": label,
                        "name": category,
                        "supercategory": supercategory,
                    }
                )

        return images, labels, file_names, categories
