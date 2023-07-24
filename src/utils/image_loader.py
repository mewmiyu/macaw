import os

from PIL import Image
from torchvision import transforms


class ImageLoader:
    def __init__(self) -> None:
        self.supercategories = ["hauptgebäude"]
        self.subcategories = ["right", "back", "left", "front"]

    def load_data(self, path_to_data):
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
                for file in files:
                    image = Image.open(os.path.join(subcategory_path, file))

                    preprocess = transforms.Compose(
                        [
                            transforms.Resize(640),
                            # transforms.CenterCrop(299),
                            transforms.ToTensor(),
                        ]
                    )
                    input_tensor = preprocess(image).to("cpu")
                    # if(input_tensor.shape != (3,299,299)):
                    #    continue
                    labels.append(label)
                    images.append(input_tensor)
                    file_names.append(file)

                label += 1
                category_parts = os.path.split(subdirs.lower())[-2:]
                supercategory = category_parts[0]
                category = "_".join([supercategory, subcategory])
                categories.append(
                    {
                        "id": label,
                        "name": category,
                        "supercategory": supercategory,
                    }
                )

        return images, labels, file_names, categories
