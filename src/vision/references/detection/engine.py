import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import vision.references.detection.utils as utils

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from vision.references.detection.coco_eval import CocoEvaluator
from vision.references.detection.coco_utils import get_coco_api_from_dataset


def train_one_epoch(
    model: FasterRCNN,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    print_freq: int,
    scaler: GradScaler = None,
    ignore_wandb: bool = False,
) -> utils.MetricLogger:
    """This function trains a model for one epoch on a dataset. The training on the
    model is done by the given optimizer on the specified device. By default weights and
    biases is used to monitor the progess. If neccessary a scaler for the gradient on
    the losses can be used.

    Args:
        model (FasterRCNN): The Faster RCNN model to be trained
        optimizer (Optimizer): The optimizer to train the model
        data_loader (DataLoader): The dataloader containing the dataset for training.
            During an epoch all images of the dataset are used to train the model
        device (str): The device on which the training is executed, one of "cuda",
            "cpu" and "mps"
        epoch (int): The number of the epoch
        print_freq (int): Describes how often the metrics are printed
        scaler (GradScaler, optional): A Grad Scaler for the gradient. Defaults to None.
        ignore_wandb (bool, optional): Whether to ignore weights and biases or to use
            it. Defaults to False.

    Returns:
        Utils.MetricLogger: The logged metrics for the training
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(
        data_loader, print_freq, header, ignore_wandb, is_train=True, epoch=epoch
    ):
        images = list(image["image"].to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        # with torch.autocast("cuda", enabled=scaler is not None):
        # loss_dict = model(images, targets)
        # losses = sum(loss for loss in loss_dict.values())
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            # Automatic Mixed Precision is not available on MPS so we just skip it
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        del targets, images

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None and torch.cuda.is_available():
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del loss_dict, loss_dict_reduced

    return metric_logger


def _get_iou_types(model: GeneralizedRCNN) -> list:
    """This functions returns the types of outputs of the model on which an
      "intersection over union" (IoU) can be calculated.

    Args:
        model (GeneralizedRCNN): The model we want to know the iou types.

    Returns:
        list: The list of iou types as strings
    """
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model: GeneralizedRCNN, data_loader: DataLoader, device: str
) -> CocoEvaluator:
    """This function evaluates a given model on the CocoEvaluation Metrics. The
    evaluation is done on the dataset, provided through the given data_loader.
    After inputting the images through the model, the coco-evaluator actually does
    the evaluation and prints it to the console.

    Args:
        model (GeneralizedRCNN): The model to be evaluated
        data_loader (DataLoader): The dataloader containing the test images
        device (str): The device on which to run the model, one of "cuda", "cpu" and
            "mps"

    Returns:
        CocoEvaluator: The Cocoevaluator, containing all the computed metrics.
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(
        data_loader, 100, header, is_train=False
    ):
        images = list(img["image"].to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        model_time = time.time()
        outputs = model(images)
        del images

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(device)
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
