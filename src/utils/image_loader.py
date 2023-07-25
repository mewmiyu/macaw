import os

from PIL import Image
from utils.preprocess import get_transform


class ImageLoader:
    def __init__(self) -> None:
        self.supercategories = ["hauptgebäude", "karo5", "piloty", "ULB"]
        self.subcategories = ["right", "back", "left", "front"]

    def __call__(self, path_to_data):
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
            supercategory_path = os.path.join(path_to_data, supercategory)
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
