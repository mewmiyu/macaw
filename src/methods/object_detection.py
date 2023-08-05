import time
import torch
import torch.utils.data
import wandb


from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from datasets.campus_dataset import CampusDataset
from vision.references.detection.engine import train_one_epoch, evaluate
from utils.preprocess import get_transform
import vision.references.detection.utils as utils


def get_object_detection_model(train_cfg: dict) -> FasterRCNN:
    """Loads and returns the faster rcnn model, which is given in the config file. The
    model is already pretrained.

    Args:
        train_cfg (dict): the training part of the configuration

    Returns:
        FasterRCNN: The pretrained detection model
    """
    # load an object detection model pre-trained on COCO
    # weights = eval(train_cfg["WEIGHTS"]).DEFAULT

    model = eval(train_cfg["META_ARCHITECTURE"])(pretrained=True)
    # get the number of input features for the classifier²
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, train_cfg["NUM_CLASSES"]
    )

    return model


def train(cfg: dict):
    """This method is used for fine-tuning the detector. The config file describes the
    parameters of the training, like the device, the batch-size, number of classes,
    architecture, etc. After the training is done, the trained model is saved to disk.

    Args:
        cfg (dict): The config file, to describe the training-process

    Raises:
        ValueError: The error is thrown if the given device is no supported
    """
    train_cfg = cfg["TRAINING"]
    device = train_cfg["DEVICE"]
    if (
        device.startswith("cuda")
        and torch.cuda.is_available()
        or device == "mps"
        and torch.backends.mps.is_available()
        or device == "cpu"
    ):
        device = torch.device(device)
    else:
        raise ValueError(
            f"We currently do not support the device: {device}, or it is not made visible to PyTorch"
        )
    num_classes = train_cfg["NUM_CLASSES"]
    test_proportion = train_cfg["TEST_PROPORTION"]
    batch_size = train_cfg["PARAMETERS"]["BATCH_SIZE"]
    num_workers = train_cfg["PARAMETERS"]["NUM_WORKERS"]

    meta_architecture = train_cfg["META_ARCHITECTURE"]
    # get the model using our helper function
    model = get_object_detection_model(train_cfg)

    # move model to the right device
    model.to(device)

    annotation_file = cfg["DATA"]["ANNOTATIONS_PATH"]
    # use our dataset and defined transformations
    dataset = CampusDataset(
        "data", annotation_file, num_classes, get_transform(train=True)
    )
    dataset_test = CampusDataset(
        "data", annotation_file, num_classes, get_transform(train=False)
    )

    # split the dataset in train and test set
    # since we apply different transformations to the train and test sets we need to
    # manually create the subsets, rather than use torch.utils.data.random_split()
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    train_set_size = int(dataset_size * (1 - test_proportion))
    train_set_size += train_set_size % batch_size
    test_set_size = dataset_size - train_set_size
    dataset = torch.utils.data.Subset(dataset, indices[:-test_set_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_set_size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    learning_rate = train_cfg["PARAMETERS"]["BASE_LR"]
    momentum = train_cfg["PARAMETERS"]["MOMENTUM"]
    weight_decay = train_cfg["PARAMETERS"]["WEIGHT_DECAY"]
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    step_size = train_cfg["PARAMETERS"]["STEPS"]
    gamma = train_cfg["PARAMETERS"]["GAMMA"]
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    num_epochs = train_cfg["PARAMETERS"]["EPOCHS"]
    dataset_parts = cfg["DATA"]["SUPERCATEGORIES"]
    if not cfg["WANDB"]["IGNORE"]:
        wandb.init(
            project=cfg["WANDB"]["PROJECT"],
            entity=cfg["WANDB"]["ENTITY"],
            config={
                "architecture": meta_architecture,
                "num_classes": num_classes,
                "epochs": num_epochs,
                "dataset": dataset_parts,
                "batch_size": batch_size,
                "optimizer": type(optimizer).__name__,
                "lr": learning_rate,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_scheduler": type(lr_scheduler).__name__,
                "step_size": step_size,
                "gamma": gamma,
            },
        )

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=10,
            ignore_wandb=cfg["WANDB"]["IGNORE"],
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch) % 10 == 0 or epoch == num_epochs - 1:
            evaluate(model, data_loader_test, device=device)
    model_file = cfg["TRAINING"]["MODEL_FILE"]
    utils.save_on_master(model, model_file)
    model_saved = torch.load(model_file)


if __name__ == "__main__":
    train({})
