import os
from typing import Any, Tuple

from PIL import Image
from utils.preprocess import get_transform


class ImageProvider:
    """The ImageProvider class is an abstraction for all image providers in our project.
    It acts as a Callable interface, forcing child classes to implement a call function.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()


class DatasetImageProvider(ImageProvider):
    """The DatasetImageProvider loads all images from the respective folders. It is an
    essential part of the labeling and training pipelines, since it creates the labels
    that are later used during training.
    """

    def __init__(self, folders: list[str], subfolders: list[str]) -> None:
        """Initialises the DatasetImageProvider class with the names of the folders and
        subfolders where the data is stored. The labels/categories are constructed in
        the format "folder_subfolder".

        Args:
            folders (str): The parent folders, later corresponding to supercategories.
            subfolders (str): The child folders, containing the images. Only images in
            theses folders are actually considered for labeling/training.
        """
        self.supercategories = folders
        self.subcategories = subfolders

    def __call__(self, data_path: str) -> Tuple:
        """Goes through the previously selected folders and subfolders inside data_path
        and loads the images it finds. The labels are created as "folder_subfolder".

        Args:
            data_path (str): The root folder for the data

        Returns:
            Tuple: images, labels and file_names are arrays, containing the
            corresponding information per image at each index. Categories contains the
            list of categories, later used as labels for the model.
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
                    image = Image.open(os.path.join(subcategory_path, file)).convert(
                        "RGB"
                    )

                    labels.append(label)
                    images.append(image)
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
