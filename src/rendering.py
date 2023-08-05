import cv2 as cv
import numpy as np

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont


DEFAULT_COLOR = (0, 255, 0)


def render_contours(img: np.ndarray, contours, color=DEFAULT_COLOR) -> np.ndarray:
    return cv.drawContours(img, [contours], 0, color, 2)


def render_fill_contours(img: np.ndarray, contours, color=DEFAULT_COLOR, alpha=0.5) -> np.ndarray:
    layer = cv.fillPoly(img.get(), [contours], color=color)
    return cv.addWeighted(img, alpha, layer, 1-alpha, 0)


def render_matches(img, kp, img2, kp2, matches):
    return cv.drawMatches(img, kp, img2, kp2, matches, None)


# function for either rendering the box with metadata or getting the result from ogre
def render_metadata(img: cv.UMat, label: str, overlays, pos=np.array((40, 40)), alpha=0.75, color=DEFAULT_COLOR):
    idx = label.rindex('_')
    name = label[:idx]
    overlay = overlays[name]

    overlay_color = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255
    alpha_mask = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))
    h, w = overlay.shape[:2]
    frame = img.get()

    # mask_offset = np.array((0, 0))

    # # Catch rendering outside the screen
    # if h + pos[0] > frame.shape[0]:
    #     h = frame.shape[0] - pos[0]
    # if w + pos[1] > frame.shape[1]:
    #     w = frame.shape[1] - pos[1]
    # if pos[0] < 0:
    #     h += pos[0]
    #     mask_offset[0] = -pos[0]
    #     pos[0] = 0
    # if pos[1] < 0:
    #     w += pos[1]
    #     mask_offset[1] = - pos[1]
    #     pos[1] = 0

    frame_subsection = frame[pos[0]:pos[0] + h, pos[1]:pos[1] + w, :]

    combine_fr_ov = frame_subsection * (1 - alpha_mask) + overlay_color * alpha_mask
    frame[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = (1 - alpha) * frame_subsection + alpha * combine_fr_ov

    return cv.UMat(frame)


def render_text(img: np.ndarray, txt: str, pos, color=DEFAULT_COLOR) -> np.ndarray:
    cv.putText(img, txt, pos, cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

