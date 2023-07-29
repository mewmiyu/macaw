import cv2 as cv
import numpy as np

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont


# function for rendering the box
## see lecture 3d transformation slide 9 for help (rendering a bunny in a known space)
## -> we need to have a callibrated camera in that example, which is not possible in our case
## we also need either a mesh or a function, which gives us the metadata
def render_img1(K, dist_coeff, vid):
    """
    Renders the square around the building
    :param K: camera callibration matrix
    :param dist_coeff:
    :param vid: a video of the building
    """
    while True:
        frame = vid.read()[1]
        ## here a function that is similar to the "detect markers function", which givs us e.g ids
        ids = None
        if ids is not None:
            ## we need to be able to compute the boundaries, then we can draw the square
            ## todo do we need edge detection rather than features? can we do both? our square needs the contours..
            ## https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            img = None
            contours = None
            cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        ## then we show the frame; theoretically, if we have a mesh we can use the function from the assignment


def render_img(img = None, ):
    if img is None:
        size = (600, 800)
        img = np.zeros(size)

    # draw all contours

    # draw all meta-data

    return img


def render_contours(img: np.ndarray, contours, color=(0, 255, 0)) -> np.ndarray:
    return cv.drawContours(img, [contours], 0, color, 2)


def render_fill_contours(img: np.ndarray, contours, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    layer = cv.fillPoly(img.get(), [contours], color=color)

    return cv.addWeighted(img, alpha, layer, 1-alpha, 0)



def render_matches(img, kp, img2, kp2, matches):
    return cv.drawMatches(img, kp, img2, kp2, matches, None)


# function for either rendering the box with metadata or getting the result from ogre
def render_metadata(img: np.ndarray) -> np.ndarray:
    # TODO: render metadata
    return img


def render_text(img: np.ndarray, txt: str) -> np.ndarray:
    cv.putText(img, txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


# from detector

def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()
):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25
        )
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii"), int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str],
            )
            np.copyto(image, np.array(image_pil))
    return image


def display_image(image):
    cv.imshow("image", image)

