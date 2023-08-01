import matplotlib.pyplot as plt
import torchvision
from collections.abc import Callable
from typing import Any, Tuple


class ImageViewer:
    """The ImageViewer provides the base functionality for showing images and bounding
    boxes. It registers basic user input, which can also be used in child classes, by
    implementing the respective function.
    """

    def __init__(self, image_provider: Callable[[], Tuple[Any, ...]]) -> None:
        """Initialises the image viewer with an image provider, which loads images,
        along with target bounding boxes and (optionally) predicted bounding boxes.

        Args:
            image_provider (Callable[[], Tuple[Any, ...]]): The image provider, which
                loads images.
        """
        self.image_provider = image_provider

        fig = plt.figure(figsize=(10, 10))
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.last_key_press = ""

    def __call__(self):
        """Loads and shows an initial image and waits for user input."""
        image, target, prediction, title = self.image_provider(silent=False)

        self.show_image(image, target, prediction, title)
        plt.show(block=True)

    def on_press(self, event):
        """Handles key presses. This function can be extended in child classes to offer
        a broader set of supported keys.
        """
        if self.last_key_press == "escape" and event.key == "escape":
            quit()
        else:
            self.last_key_press = event.key

        if event.key == "n":
            image, target, prediction, title = self.image_provider(silent=False)
            self.show_image(image, target, prediction, title=title)

    def on_click(self, event):
        """Handles mouse clicks. This function can and should be extended in a child
        class, if support for mouse clicks is needed.
        """
        pass

    def show_image(self, image, target, prediction=None, title=""):
        plt.clf()
        plt.imshow(image, zorder=1)
        plt.title(title)

        if target is not None:
            print(f"[INFO] Annotation:")
            self.draw_bbox(*target[0], format="XYXY")

        if prediction is None:
            return

        print(f"[INFO] Predictions (out of {len(prediction['boxes'])}):")
        # loop over the detections
        for j, bbox in enumerate(prediction["boxes"]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = prediction["scores"][j]
            # filter out weak detections by ensuring the confidence is greater than a
            # certain threshold
            if confidence > 0.7:
                print(f"[INFO] Confidence: {confidence}")
                # extract the index of the class label from the detections, then compute
                # the (x, y)-coordinates of the bounding box for the object
                idx = int(prediction["labels"][j])
                label = "{}: {:.2f}%".format(
                    self.image_provider.category_labels[idx], confidence * 100
                )
                # display the prediction to our terminal
                print("[INFO] {}".format(label))

                # draw the bounding box and label of the image
                bbox = bbox.detach().cpu().numpy()
                self.draw_bbox(*bbox, format="XYXY", color="red")

                y = bbox[3] - 15 if bbox[3] - 15 > 15 else bbox[3] + 15
                plt.text(bbox[0], y, label, color="red")

    def draw_bbox(
        self,
        minx: float,
        miny: float,
        width: float,
        height: float,
        format: str = "XYWH",
        color: str = "blue",
    ):
        """This function draws a bounding box onto the current matplotlib plot. The
        bounding box itself is described through four values, minx and miny describing
        the top-left corner of the bounding box. Based on the format, width and height
        are either the actual width and height of the bounding box, or the bottom-right
        corner. The last parameter describes the color of the bounding box.

        Args:
            minx (float): smallest x-value of the bounding box
            miny (float): smallest y-value of the bounding box
            width (float): Either biggest x-value of the bounding box or width
            height (float): Either biggest y-value of the bounding box or height
            format (str, optional): The format of the provided bounding box data. Either
                "XYWH" or "XYXY". Defaults to "XYWH".
            color (str, optional): Color of the bounding box. Defaults to "blue".

        Raises:
            ValueError: Raised if the format of the bounding box data is not valid.
        """
        if format == "XYWH":
            maxx = minx + width
            maxy = miny + height
        elif format == "XYXY":
            maxx = width
            maxy = height
        else:
            raise ValueError("We currently only support XYWH and XYXY formats.")

        print(f"[INFO] Bounding Box: {(miny, minx, maxy, maxx)}")
        # cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
        plt.plot([minx, minx], [miny, maxy], color, zorder=2)
        plt.plot([maxx, maxx], [miny, maxy], color, zorder=2)
        plt.plot([minx, maxx], [miny, miny], color, zorder=2)
        plt.plot([minx, maxx], [maxy, maxy], color, zorder=2)
        plt.draw()
