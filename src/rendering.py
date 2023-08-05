import cv2 as cv
import numpy as np


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

    frame_subsection = frame[pos[0]:pos[0] + h, pos[1]:pos[1] + w, :]

    combine_fr_ov = frame_subsection * (1 - alpha_mask) + overlay_color * alpha_mask
    frame[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = (1 - alpha) * frame_subsection + alpha * combine_fr_ov

    return cv.UMat(frame)


def render_text(img: np.ndarray, txt: str, pos, color=DEFAULT_COLOR) -> np.ndarray:
    cv.putText(img, txt, pos, cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def display_image(image):
    cv.imshow("image", image)

