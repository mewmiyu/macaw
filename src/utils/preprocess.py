import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T


def get_transform(train: bool) -> T.Compose:
    """This function returns the transformations for the datasets. This includes a
    conversion to a Tensor, as well as a resize. If train is set to true,
    data-augmentation can be applied, which is disabled for now.

    Args:
        train (bool): Adds data augmentation to the transformation

    Returns:
        Transform: The combined transformation to be used on the dataset
    """
    transforms = []

    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToImageTensor())
    transforms.append(T.ConvertImageDtype())
    transforms.append(T.Resize(640, antialias=True))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return T.Compose(transforms)
