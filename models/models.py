
import segmentation_models_pytorch as smp
from .fpn import resnet34_fpn, effnetB4_fpn


def get_model(model_type: str = 'Unet',
              encoder: str = 'Resnet18',
              encoder_weights: str = 'imagenet',
              activation: str = None,
              n_classes: int = 4):
    """
    # https://github.com/BloodAxe/Kaggle-2018-Airbus/tree/master/lib/models

    :param model_type:
    :param encoder:
    :param encoder_weights:
    :param activation:
    :param n_classes:
    :return:
    """
    if model_type == 'Unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=n_classes,
            activation=activation
        )

    elif model_type == 'Linknet':
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=n_classes,
            activation=activation
        )

    elif model_type == 'resnet34_fpn':
        model = resnet34_fpn(num_classes=n_classes, fpn_features=128)

    elif model_type == 'effnetB4_fpn':
        model = effnetB4_fpn(num_classes=n_classes, fpn_features=128)

    else:
        model = None

    return model
