import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torchvision.transforms as T

from datasets.campus_dataset import CampusDataset
import methods.torchvision_utils as utils


def get_transform(train):
    transforms = []
    transforms.append(T.Resize(640))
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        #transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return T.Compose(transforms)


class Viewer:
    def __init__(self) -> None:
        self.model = torch.load("faster_rcnn-working-epoch.pt").to("cuda")
        self.model.eval()

        dataset = CampusDataset("annotations_full.json", get_transform(train=True))
        self.category_labels = {
            cat["id"]: cat["name"] for cat in dataset.categories.values()
        }
        batch_size = 1
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

    def run_inference(self):
        self.data, targets = next(iter(self.data_loader))
        self.images = list(image["image"].to("cuda") for image in self.data)
        self.image_names = list(image["filename"] for image in self.data)
        targets = [{k: v for k, v in t.items()} for t in targets]
        self.title = self.image_names[0]
        start_time = time.time()
        self.predictions = self.model(self.images)  # Returns predictions
        inference_time = time.time() - start_time
        print("Inference time", inference_time)

        return np.array(self.images[0].detach().to("cpu").permute((1, 2, 0))), targets[0]['boxes']

    def __call__(self):
        image, target = self.run_inference()

        fig = plt.figure(figsize=(10, 10))
        fig.canvas.mpl_connect("key_press_event", self.on_press)
        # plt.title(self.title)
        self.show_image(image, target)
        plt.show(block=True)

    def on_press(self, event):
        # plt.close()
        if event.key == "n":
            image, target = self.run_inference()
            self.show_image(image, target)

    def show_image(self, image, target):
        i = 0
        plt.clf()
        plt.imshow(image, zorder=1)
        plt.title(self.title)
        print(f"Annotation: {target[0]}")
        (minx, miny, width, height) = target[0]
        maxx = minx + width
        maxy = miny + height
        plt.plot([minx, minx], [miny, maxy], "blue", zorder=2)
        plt.plot([maxx, maxx], [miny, maxy], "blue", zorder=2)
        plt.plot([minx, maxx], [miny, miny], "blue", zorder=2)
        plt.plot([minx, maxx], [maxy, maxy], "blue", zorder=2)
        plt.draw()
        # loop over the detections
        for j, bbox in enumerate(self.predictions[i]["boxes"]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = self.predictions[i]["scores"][j]
            print(f"Confidence: {confidence}")
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.7:
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(self.predictions[i]["labels"][j])
                bbox = bbox.detach().cpu().numpy()
                (minx, miny, width, height) = bbox.astype("int")
                multiplication_factor = 1  # (3000 / 640)
                minx = minx * multiplication_factor
                miny = miny * multiplication_factor
                maxx = minx + width * multiplication_factor
                maxy = miny + height * multiplication_factor
                print(f"[INFO] Bounding Box in int: {(miny, minx, maxy, maxx)}")

                plt.plot([minx, minx], [miny, maxy], "red", zorder=2)
                plt.plot([maxx, maxx], [miny, maxy], "red", zorder=2)
                plt.plot([minx, maxx], [miny, miny], "red", zorder=2)
                plt.plot([minx, maxx], [maxy, maxy], "red", zorder=2)
                plt.draw()

                label = "{}: {:.2f}%".format(
                    self.category_labels[idx], confidence * 100
                )
                # display the prediction to our terminal
                print("[INFO] {}".format(label))
                print(f"[INFO] Bounding-Box: {bbox}")
                # # draw the bounding box and label on the image
                # cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = maxy - 15 if maxy - 15 > 15 else maxy + 15
                plt.text(minx, y, label, color="red")
