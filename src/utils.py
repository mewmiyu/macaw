from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import yaml

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