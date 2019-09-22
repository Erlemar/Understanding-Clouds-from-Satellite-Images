import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback
# from catalyst.contrib.criterion.lovasz import LovaszLossMultiClass, LovaszLossBinary
import segmentation_models_pytorch as smp
import datetime
import argparse
import warnings
import gc
from dataset import prepare_loaders
from models.models import get_model
from optimizers import get_optimizer
from utils import get_optimal_postprocess
from predict import predict
from losses.losses import FocalLoss, BCEMulticlassDiceLoss
from losses.lovasz_losses import lovasz_softmax
warnings.filterwarnings("once")


if __name__ == '__main__':
    """
    Example of usage:
    >>> python train.py --chunk_size=10000 --n_jobs=10

    """

    parser = argparse.ArgumentParser(description="Train model for understanding_cloud_organization competition")
    parser.add_argument("--path", help="path to files", type=str, default='f:/clouds')
    # https://github.com/qubvel/segmentation_models.pytorch
    parser.add_argument("--encoder", help="u-net encoder", type=str, default='se_resnext50_32x4d')
    parser.add_argument("--encoder_weights", help="pre-training dataset", type=str, default='imagenet')
    parser.add_argument("--DEVICE", help="device", type=str, default='CUDA')
    parser.add_argument("--num_workers", help="num_workers", type=int, default=0)
    parser.add_argument("--bs", help="batch size", type=int, default=4)
    parser.add_argument("--lr", help="learning rate", type=int, default=1e-3)
    parser.add_argument("--lr_e", help="learning rate for decoder", type=int, default=1e-3)
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("--separate_decoder", help="number of epochs", type=bool, default=True)
    parser.add_argument("--scheduler", help="scheduler", type=str, default='ReduceLROnPlateau')
    parser.add_argument("--loss", help="loss", type=str, default='BCEDiceLoss')
    parser.add_argument("--multigpu", help="use multi-gpu", type=bool, default=True)
    parser.add_argument("--gradient_accumulation", help="gradient_accumulation steps", type=int, default=None)
    parser.add_argument("--optimize_postprocess", help="to optimize postprocess", type=bool, default=True)
    parser.add_argument("--make_prediction", help="to make prediction", type=bool, default=True)

    args = parser.parse_args()

    sub_name = f'Model_{args.encoder}_bs_{args.bs}_{str(datetime.datetime.now().date())}'
    logdir = f"./logs/{sub_name}"

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    loaders = prepare_loaders(path=args.path, bs=args.bs,
                              num_workers=args.num_workers, preprocessing_fn=preprocessing_fn)
    test_loader = loaders['test']
    del loaders['test']

    model = get_model(model_type='Unet', encoder=args.encoder, encoder_weights=args.encoder_weights, activation=None)

    optimizer = get_optimizer(optimizer='RAdam', lookahead=False, model=model,
                              separate_decoder=args.separate_decoder, lr=args.lr, lr_e=args.lr_e)

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=3)
    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=3)

    if args.loss == 'BCEDiceLoss':
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    elif args.loss == 'BCEJaccardLoss':
        criterion = smp.utils.losses.BCEJaccardLoss(eps=1.)
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss()
    # elif args.loss == 'lovasz_softmax':
    #     criterion = lovasz_softmax()
    elif args.loss == 'BCEMulticlassDiceLoss':
        criterion = BCEMulticlassDiceLoss()
    else:
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)

    runner = SupervisedRunner()

    if args.multigpu:
        model = nn.DataParallel(model)

    callbacks = [DiceCallback(), EarlyStoppingCallback(patience=7, min_delta=0.001), CriterionCallback()]

    if args.gradient_accumulation:
        callbacks.append(OptimizerCallback(accumulation_steps=args.gradient_accumulation))

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=args.num_epochs,
        verbose=True
    )

    torch.cuda.empty_cache()
    gc.collect()

    if args.optimize_postprocess:
        class_params = get_optimal_postprocess(loaders, runner, logdir)

    if args.make_prediction:
        loaders['test'] = test_loader
        predict(loaders=loaders, runner=runner, class_params=class_params, path=args.path, sub_name=sub_name)
