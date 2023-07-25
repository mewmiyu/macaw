import json
import os
import torch.utils.data

from PIL import Image


class CampusDataset(torch.utils.data.Dataset):
    def __init__(self, root="", transforms=None):
        self.root = root
        self.transforms = transforms
        with open("annotations_full.json", mode="r") as f:
            annotations = json.load(f)
        keys = ["categories", "images", "annotations"]
        if any([key not in annotations for key in keys]):
            raise ValueError("Annotations file is in the wrong format.")

        self.categories, self.imgs, self.annotations = annotations.values()
        self.categories = {c["id"]: {**c, "id": c["id"] + 1} for c in self.categories}
        self.imgs = {i["id"]: i for i in self.imgs}

    def __getitem__(self, index):
        id, img_id, ct_id, bbox, area, seg, iscrowd = self.annotations[index].values()

        category = self.categories[ct_id]

        boxes = torch.as_tensor([bbox], dtype=torch.float32)
        boxes[:, 2] = boxes[:, 0] +  boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] +  boxes[:, 3]
        #print(boxes.shape)
        labels = torch.as_tensor([category["id"]], dtype=torch.int64)
        iscrowd = torch.as_tensor([iscrowd], dtype=torch.int64)
        area = torch.as_tensor([area], dtype=torch.int64)

        path_parts = category["name"].split("_")
        img_path = os.path.join("data", *path_parts, self.imgs[img_id]["file_name"])
        img = Image.open(img_path)

        target = {
            "image_id": torch.as_tensor(img_id, dtype=torch.int64),
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "area": area,
        }

        if self.transforms is not None:
            # TODO: See how we can apply transformations to the annotations
            # img, target = self.transforms(img, target) # This did not work
            img = self.transforms(img)

        return {"image": img, "filename": self.imgs[img_id]["file_name"]}, target

    def __len__(self):
        return len(self.imgs)
