from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import yaml

import tempfile
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


# From detector: #TODO: Rendering stuff into renderer!
def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.show()


def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    # response = urlopen(url)
    # image_data = response.read()
    # image_data = BytesIO(image_data)
    pil_image = Image.open(url)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


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


def crop_img(img:np.ndarray((-1, -1, 3)), min_:np.ndarray((2,), dtype=int), max_:np.ndarray((2,), dtype=int)) -> np.ndarray((-1, -1, 3)) :

    return img











"""
    This function loads all images from the data directory. 
    Each new directory creates a new label, so images from
    the same category should be in the same directory
"""
def load_data(path_to_data):
    labels = []
    images = []
    label = -1
    subdir = ''
    for subdirs, _, files in os.walk(path_to_data):
        if(subdirs != 'data'):
            for file in files:
                if(subdirs != subdir):
                    label += 1
                    subdir = subdirs
                image = Image.open(os.path.join(subdirs, file))
                preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(-127.5, 127.5)
                ])
                input_tensor = preprocess(image).to('cuda')
                if(input_tensor.shape != (3,299,299)):
                    continue
                labels.append(label)
                images.append(input_tensor)
    return images, np.array(labels)

"""
    This function generates a dataset based on triplets. It returns
    a numpy array of size [batch_amount, batch_size, 3]. Each entry 
    is an index describing the position of the data based on the
    labels input
"""
def gen_triplet_dataset(labels, batch_size, batch_amount):
    dataset = []
    max_label = np.max(labels)
    for _ in range(batch_amount):
        batch = []
        for b in range(batch_size):
            label1 = np.random.randint(0, max_label+1)
            label2 = np.random.randint(0, max_label+1)
            while label1 == label2:
                label2 = np.random.randint(0, max_label+1)
            label1_pos = np.where(labels == label1)
            l1_min = np.min(label1_pos)
            l1_max = np.max(label1_pos)
            label2_pos = np.where(labels == label2)
            l2_min = np.min(label2_pos)
            l2_max = np.max(label2_pos)
            anchor = np.random.randint(l1_min, l1_max+1)
            positive = np.random.randint(l1_min, l1_max+1)
            while positive == anchor and l1_min != l1_max:
                positive = np.random.randint(l1_min, l1_max+1)
            negative = np.random.randint(l2_min, l2_max+1)
            batch.append([anchor, positive, negative])
        dataset.append(np.array(batch))
    return np.array(dataset)

"""
    Reads in a yaml config file from a filepath
"""
def read_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data