import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from typing import Any, Tuple

import vision.references.detection.utils as utils
from datasets.campus_dataset import CampusDataset
from utils.image_loader import ImageProvider
from utils.preprocess import get_transform


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
        self.data, targets = next(iter(self.data_loader))
        self.images = list(image["image"].to(self.device) for image in self.data)
        self.image_names = list(image["filename"] for image in self.data)
        targets = [{k: v for k, v in t.items()} for t in targets]

        start_time = time.time()
        predictions = self.model(self.images)
        inference_time = time.time() - start_time

        if not silent:
            print("[INFO] Inference time", inference_time)

        return (
            np.array(self.images[0].detach().to("cpu").permute((1, 2, 0))),
            targets[0]["boxes"],
            predictions[0],
            self.image_names[0],
        )
