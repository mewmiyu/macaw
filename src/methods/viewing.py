import matplotlib.pyplot as plt
import torchvision
from collections.abc import Callable
from typing import Any, Tuple


class ImageViewer:
    def __init__(self, image_provider: Callable[[], Tuple[Any, ...]]) -> None:
        self.image_provider = image_provider

        fig = plt.figure(figsize=(10, 10))
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.last_key_press = ""

    def __call__(self):
        image, target, prediction, title = self.image_provider()

        fig = plt.figure(figsize=(10, 10))
        fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.show_image(image, target, prediction, title)
        plt.show(block=True)

    def on_press(self, event):
        if self.last_key_press == "escape" and event.key == "escape":
            quit()
        else:
            self.last_key_press = event.key

        if event.key == "n":
            image, target, prediction = self.image_provider()
            self.show_image(image, target, prediction)

    def on_click(self, event):
        pass

    def show_image(self, image, target, prediction=None, title=""):
        plt.clf()
        plt.imshow(image, zorder=1)
        plt.title(title)

        if target is not None:
            print(f"[INFO] Annotation:")
            self.draw_bbox(*target[0])

        if prediction is None:
            return

        print(f"[INFO] Predictions (out of {len(prediction['boxes'])}):")
        # loop over the detections
        for j, bbox in enumerate(prediction["boxes"]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = prediction["scores"][j]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.7:
                print(f"[INFO] Confidence: {confidence}")
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(prediction["labels"][j])
                label = "{}: {:.2f}%".format(
                    self.image_provider.category_labels[idx], confidence * 100
                )
                # display the prediction to our terminal
                print("[INFO] {}".format(label))

                # draw the bounding box and label on the image
                bbox = bbox.detach().cpu().numpy()
                self.draw_bbox(*bbox, format="XYXY", color="red")

                y = bbox[3] - 15 if bbox[3] - 15 > 15 else bbox[3] + 15
                plt.text(bbox[0], y, label, color="red")

    def draw_bbox(self, minx, miny, width, height, format="XYWH", color="blue"):
        if format == "XYWH":
            maxx = minx + width
            maxy = miny + height
        elif format == "XYXY":
            maxx = width
            maxy = height
        else:
            raise ValueError("We only currently only support XYWH and XYXY formats")

        print(f"[INFO] Bounding Box: {(miny, minx, maxy, maxx)}")
        # cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
        plt.plot([minx, minx], [miny, maxy], color, zorder=2)
        plt.plot([maxx, maxx], [miny, maxy], color, zorder=2)
        plt.plot([minx, maxx], [miny, miny], color, zorder=2)
        plt.plot([minx, maxx], [maxy, maxy], color, zorder=2)
        plt.draw()
