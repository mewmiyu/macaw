import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T


def get_transform(train: bool):
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
