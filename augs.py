import albumentations as albu


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_training_augmentation(augmentation: str='default', image_size: tuple = (320, 640)):
    """

    Args:
        augmentation:
        image_size:

    Returns:

    """
    LEVELS = {
        'default': get_training_augmentation0,
        '1': get_training_augmentation1,
        '2': get_training_augmentation2
    }

    assert augmentation in LEVELS.keys()
    return LEVELS[augmentation](image_size)


def get_training_augmentation0(image_size: tuple = (320, 640)):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.5),
        albu.RandomGamma(),
        albu.Resize(*image_size)
    ]
    return albu.Compose(train_transform)


def get_training_augmentation1(image_size: tuple = (320, 640)):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.Resize(*image_size),
    ]
    return albu.Compose(train_transform)


def get_training_augmentation2(image_size: tuple = (320, 640)):
    train_transform = [
        albu.Resize(*image_size),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.Blur(),
        albu.RandomBrightnessContrast()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size: tuple = (320, 640)):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(*image_size)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
