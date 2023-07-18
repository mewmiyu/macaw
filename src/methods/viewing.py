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
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class Viewer:
    def __init__(self) -> None:
        self.model = torch.load("faster_rcnn.pt").to("mps")
        self.model.eval()

        dataset = CampusDataset("annotations_full.json", get_transform(train=True))
        self.category_labels = {
            cat["id"]: cat["name"] for cat in dataset.categories.values()
        }
        batch_size = 2
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn,
        )

    def run_inference(self):
        self.images, targets = next(iter(self.data_loader))
        self.images = list(image.to("mps") for image in self.images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        start_time = time.time()
        self.predictions = self.model(self.images)  # Returns predictions
        inference_time = time.time() - start_time
        print("Inference time", inference_time)

        return np.array(self.images[0].detach().to("cpu").permute((1, 2, 0)))

    def __call__(self):
        image = self.run_inference()

        fig = plt.figure(figsize=(10, 10))
        fig.canvas.mpl_connect("key_press_event", self.on_press)
        # plt.title(self.title)
        self.show_image(image)
        plt.show(block=True)

    def on_press(self, event):
        # plt.close()
        if event.key == "n":
            image = self.run_inference()
            self.show_image(image)

    def show_image(self, image):
        i = 0
        plt.clf()
        # loop over the detections
        for j, bbox in enumerate(self.predictions[i]["boxes"]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = self.predictions[i]["scores"][j]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.4:
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(self.predictions[i]["labels"][j])
                bbox = bbox.detach().cpu().numpy()
                (minx, miny, width, height) = bbox.astype("int")
                maxx = minx + width
                maxy = miny + height

                plt.imshow(image, zorder=1)

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
                # # draw the bounding box and label on the image
                # cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = maxy - 15 if maxy - 15 > 15 else maxy + 15
                plt.text(minx, y, label, color="red")
