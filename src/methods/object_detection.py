import math
import time
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torch.distributed as dist
import sys
import wandb

from datasets.campus_dataset import CampusDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from methods.torchvision_engine import train_one_epoch, evaluate
import methods.torchvision_utils as utils


def get_object_detection_model(num_classes=3):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(T.Resize(640))
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return T.Compose(transforms)


def train_experiment_fn(cfg):
    model = get_object_detection_model(num_classes=3)
    dataset = CampusDataset("annotations_full.json", get_transform(train=True))
    batch_size = 2
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image.to("mps") for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    # targets = [
    #     {key: value[i] for key, value in targets.items()} for i in range(batch_size)
    # ]

    model.to("mps")
    output = model(images, targets)  # Returns losses and detections
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400, device="mps"), torch.rand(3, 500, 400, device="mps")]
    start_time = time.time()
    predictions = model(x)  # Returns predictions
    inference_time = time.time() - start_time
    print("Inference time", inference_time)


def train(cfg):
    # train on the GPU or on the CPU, if a GPU is not available
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(device)

    # our dataset has two classes only - background and person
    num_classes = 3
    # use our dataset and defined transformations
    dataset = CampusDataset("annotations_full.json", get_transform(train=True))
    # subsets = torch.utils.data.random_split(dataset, [0.8, 0.2])
    dataset_test = CampusDataset("annotations_full.json", get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-14])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-14:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 50
    wandb.init(
        project="augmented-vision",
        entity="macaw",
        config={
            "architecture": "fasterrcnn_resnet50_fpn",
            "num_classes": num_classes,
            "epochs": num_epochs,
            "dataset": "hauptgebäude",
            "batch_size": 2,
            "optimizer": "SGD",
            "lr": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "lr_scheduler": "StepLR",
            "step_size": 20,
            "gamma": 0.1,
        },
    )

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch + 1) % 10 == 0:
            evaluate(model, data_loader_test, device=device)

    utils.save_on_master(model, "faster_rcnn-50-epoch.pt")
    model_saved = torch.load("faster_rcnn-50-epoch.pt")
    print("That's it!")


if __name__ == "__main__":
    train({})
