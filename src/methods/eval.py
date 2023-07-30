import numpy as np
import time
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from typing import Any, Tuple

import vision.references.detection.utils as utils
from datasets.campus_dataset import CampusDataset
from src.utils.image_loader import ImageProvider
from src.utils.preprocess import get_transform


class TorchImageProvider(ImageProvider):
    def __init__(
        self,
        annotations,
        model_checkpoint=None,
        device="cpu",
        batch_size=1,
        num_workers=2,
    ) -> None:
        super().__init__()

        self.device = device
        if model_checkpoint is not None:
            self.model = torch.load(model_checkpoint, map_location=self.device)
            self.model.eval()
        else:
            raise ValueError(f"No model checkpoint was provided!")

        dataset = CampusDataset(annotations, get_transform(train=False))
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

    def __call__(self, silent=True) -> Tuple[Any, ...]:
        data, targets = next(iter(self.data_loader))
        images = list(image["image"].to(self.device) for image in data)
        image_names = list(image["filename"] for image in data)
        targets = [{k: v for k, v in t.items()} for t in targets]

        start_time = time.time()
        predictions = self.model(images)
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
    def __init__(self, model_checkpoint=None, annotations=None, device="cpu") -> None:
        super().__init__()

        self.device = device
        if model_checkpoint is not None:
            self.model = torch.load(model_checkpoint, map_location=self.device)
            self.model.eval()
        else:
            raise ValueError(f"No model checkpoint was provided!")

        if annotations is not None:
            dataset = CampusDataset(annotations, get_transform(train=False))
            self.category_labels = {
                cat["id"]: cat["name"] for cat in dataset.categories.values()
            }

    def __call__(self, image: NDArray[np.uint8], silent=True):
        image_float = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_float).to(self.device).permute((2, 0, 1))
        self.images = [image_tensor]

        start_time = time.time()
        predictions = self.model(self.images)
        inference_time = time.time() - start_time

        if not silent:
            print("[INFO] Inference time", inference_time)

        return (
            np.array(self.images[0].detach().to("cpu").permute((1, 2, 0))),
            None,
            predictions[0],
            None,
        )
