import cv2 as cv
import numpy as np


DEFAULT_COLOR = (0, 255, 0)


def render_contours(img: np.ndarray, contours, color=DEFAULT_COLOR) -> np.ndarray:
    """
    Renders the contours on the image.

    Args:
        img (np.ndarray): The image to render the contours on.
        contours (np.ndarray): The contours to render.
        color (tuple, optional): The color of the contours. Defaults to DEFAULT_COLOR.

    Returns:
        np.ndarray: The image with the contours rendered on it.
    """
    return cv.drawContours(img, [contours], 0, color, 2)


def render_fill_contours(img: np.ndarray, contours, color=DEFAULT_COLOR, alpha=0.5) -> np.ndarray:
    """
    Renders the filled contours on the image.

    Args:
        img (np.ndarray): The image to render the contours on.
        contours (np.ndarray): The contours to render.
        color (tuple, optional): The color of the contours. Defaults to DEFAULT_COLOR.
        alpha (float, optional): The alpha value of the fill. Defaults to 0.5.

    Returns:
        np.ndarray: The image with the filled contours rendered on it.
    """
    layer = cv.fillPoly(img.get(), [contours], color=color)
    return cv.addWeighted(img, alpha, layer, 1-alpha, 0)


def render_matches(img, kp, img2, kp2, matches):
    """
    Renders the matches between two images.

    Args:
        img (np.ndarray): The first image.
        kp (np.ndarray): The keypoints of the first image.
        img2 (np.ndarray): The second image.
        kp2 (np.ndarray): The keypoints of the second image.
        matches (np.ndarray): The matches between the two images.

    Returns:
        np.ndarray: The image with the matches rendered on it.
    """
    return cv.drawMatches(img, kp, img2, kp2, matches, None)


def render_metadata(img: cv.UMat, label: str, overlays, pos=np.array((40, 40)), alpha=0.75):
    """
    Renders the overlay with the metadata on the image.

    Args:
        img (cv.UMat): The image to render the overlay on.
        label (str): The label of the overlay.
        overlays (dict): The overlays to render.
        pos (np.array, optional): The position of the overlay. Defaults to np.array((40, 40)).
        alpha (float, optional): The alpha value for the overlay. Defaults to 0.75.

    Returns:
        cv.UMat: The image with the overlay rendered on it.
    """
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
    """
    Renders the text on the image.

    Args:
        img (np.ndarray): The image to render the text on.
        txt (str): The text to render.
        pos (tuple): The position of the text.
        color (tuple, optional): The color of the text. Defaults to DEFAULT_COLOR.

    Returns:
        np.ndarray: The image with the text rendered on it.
    """
    cv.putText(img, txt, pos, cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def display_image(image):
    """
    Displays the image.

    Args:
        image (np.ndarray): The image to display.

    Returns:
        None
    """
    cv.imshow("image", image)

