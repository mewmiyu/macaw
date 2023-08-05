import numpy as np
import time
import torch
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import DataLoader
from typing import Any, Tuple

import vision.references.detection.utils as utils
from datasets.campus_dataset import CampusDataset
from utils.image_loader import ImageProvider
from utils.preprocess import get_transform


class TorchImageProvider(ImageProvider):
    """The TorchImageProvider uses a DataLoader to load random images from our dataset
    and runs inference on them, using the provided model.
    """

    def __init__(
        self,
        root: str,
        annotations: str,
        num_classes: int,
        model_checkpoint: str = None,
        device: str = "cpu",
        batch_size: str = 1,
        num_workers: str = 2,
    ) -> None:
        """Initialises the TorchImageProvider with the path to the annotations, which
        are used to create our CampusDataset, from which the random images are drawn
        using torch.utils.data.DataLoader.

        Args:
            root (str): Path to the dataset.
            annotations (str): Path to the annotations file.
            num_classes (int): Number of classes used during training.
            model_checkpoint (str, optional): Path to the model checkpoint, if none is
                provided, no predictions are made. Defaults to None.
            device (str, optional): The device on which to run the model, one of "cuda",
                "cpu" and "mps". Defaults to "cpu".
            batch_size (int, optional): Number of images to run inference on. Defaults
                to 1.
            num_workers (int, optional): Number of workers to use for loading data.
                Defaults to 2.
        """
        super().__init__()

        self.device = device
        if model_checkpoint is not None:
            self.model = torch.load(model_checkpoint, map_location=self.device)
            self.model.eval()

        dataset = CampusDataset(
            root, annotations, num_classes, get_transform(train=False)
        )
        self.category_labels = {
            cat["id"]: cat["name"] for cat in dataset.categories.values()
        }
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=utils.collate_fn,
        )

    def __call__(self, silent=True) -> Tuple[NDArray, ArrayLike, ArrayLike, str]:
        """Loads the next image from the DataLoader and runs inference on it, if a model
        checkpoint was provided.

        Args:
            silent (bool, optional): Whether or not to print the inference time.
                Defaults to True.

        Returns:
            Tuple[Any, ...]: The image, true box, predicted box and image name
        """
        data, targets = next(iter(self.data_loader))
        images = list(image["image"].to(self.device) for image in data)
        image_names = list(image["filename"] for image in data)
        targets = [{k: v for k, v in t.items()} for t in targets]

        start_time = time.time()
        if self.model:
            predictions = self.model(images)
        else:
            predictions = [None]
        inference_time = time.time() - start_time

        if not silent:
            print("[INFO] Inference time", inference_time)

        return (
            np.array(images[0].detach().to("cpu").permute((1, 2, 0))),
            targets[0]["boxes"],
            predictions[0],
            image_names[0],
        )


class PredictionsProvider(ImageProvider):
    """The TorchImageProvider uses a DataLoader to load random images from our dataset
    and runs inference on them, using the provided model.
    """

    def __init__(
        self,
        root: str,
        annotations: str,
        num_classes: int,
        model_checkpoint: str,
        device: str = "cpu",
        queue_size: int = 10,
    ) -> None:
        """Initialises the PredictionsProvider with a path to the model checkpoint and
        another path, to the annotation file.

        Args:
            root (str): Path to the dataset.
            annotations (str, optional): Path to the annotations file.
            num_classes (int): Number of classes used during training.
            model_checkpoint (str): Path to the model checkpoint.
            device (str, optional): The device on which to run the model, one of "cuda",
                "cpu" and "mps". Defaults to "cpu".
            queue_size (int, optional): Size of the queue to store the last predictions

        Raises:
            ValueError: _description_
        """
        super().__init__()

        self.device = device
        self.queue = []
        self.queue_size = queue_size
        if model_checkpoint is not None:
            self.model = torch.load(model_checkpoint, map_location=self.device)
            self.model.eval()
        else:
            raise ValueError(f"The path to the model checkpoint was not provided!")

        if annotations is not None:
            dataset = CampusDataset(
                root, annotations, num_classes, get_transform(train=False)
            )
            self.category_labels = {
                cat["id"]: cat["name"] for cat in dataset.categories.values()
            }
        else:
            raise ValueError(f"The path to the annotations was not provided!")

    def __call__(
        self, image: NDArray[np.uint8], silent: bool = True
    ) -> Tuple[bool, list, str, float]:
        """Runs inference on the input image and maps the predicted label to a string
        from the annotation file.

        Args:
            image (NDArray[np.uint8]): The input image, with values in range [0; 255]
            silent (bool, optional): Whether or not to print the inference time.
                Defaults to True.

        Returns:
            Tuple[bool, list, str, float]: Boolean showing if there were any predicted
                boxes, the box, label and score of the best prediction.
        """
        image_float = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_float).to(self.device).permute((2, 0, 1))
        self.images = [image_tensor]

        start_time = time.time()
        predictions = self.model(self.images)
        inference_time = time.time() - start_time

        score_best = 0.0
        if len(predictions[0]["boxes"]) > 0:
            score_best = predictions[0]["scores"][0].item()
        if len(predictions[0]["boxes"]) > 0 and score_best > 0.70:
            bbox_best = np.array(
                predictions[0]["boxes"][0].detach().to("cpu"), dtype=np.int32
            )
            label_best = self.category_labels[predictions[0]["labels"][0].item()]
            res = (True, bbox_best, label_best, score_best)

            log_msg = f"[INFO] Inference time: {inference_time} | {label_best} | Confidence: {score_best} | Box: {bbox_best}"
        else:
            res = (False, [0, 0, 0, 0], None, None)
            log_msg = f"[INFO] Inference time: {inference_time}"

        if not silent:
            print(log_msg)

        if len(self.queue) == self.queue_size:
            self.queue.pop(0)

        # self.queue.append(res)
        # labels = [item[2] for item in self.queue]
        # unique, count = np.unique(labels, return_counts=True)
        # label = unique[np.argmax(count)]
        # print(self.queue)
        # print(len(self.queue))
        # res[2] = label
        return res
