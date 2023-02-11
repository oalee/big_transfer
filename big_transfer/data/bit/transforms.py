import torchvision as tv
from timm.data.auto_augment import augment_and_mix_transform, auto_augment_transform


def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96 * 96 else (512, 480)


def get_transforms(img_size: list):

    precrop, crop = get_resolution(img_size)
    train_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((precrop, precrop)),
            tv.transforms.RandomCrop((crop, crop)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((crop, crop)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return train_transform, val_transform


def timm_transforms(image_size, transform):

    precrop, crop = get_resolution(image_size)
    train_transform = tv.transforms.Compose(
        [
            transform,
            tv.transforms.Resize((precrop, precrop)),
            tv.transforms.RandomCrop((crop, crop)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((crop, crop)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return train_transform, val_transform
