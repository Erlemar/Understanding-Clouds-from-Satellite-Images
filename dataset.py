import os
import cv2


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
import warnings
from augs import get_training_augmentation, get_training_augmentation1, get_validation_augmentation, get_preprocessing
warnings.filterwarnings("once")


def get_img(x, folder: str = 'train_images'):
    """
    Return image based on image name and folder.
    """
    image_path = os.path.join(folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class CloudDataset(Dataset):

    def __init__(self, path: str = '',
                 df: pd.DataFrame = None,
                 datatype: str = 'train',
                 img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip()]),
                 preprocessing=None,
                 preload=False):
        """

        Args:
            path:
            df:
            datatype:
            img_ids:
            transforms:
            preprocessing:
            preload:
        """

        self.df = df
        self.path = path
        self.datatype = datatype if datatype == 'test' else 'train'
        if self.datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        if self.preprocessing.__name__ == 'get_training_augmentation1':
            self.dir_name = f"{self.path}/preload_1_{self.transforms[-1].height}_{self.transforms[-1].width}"
        else:
            self.dir_name = f"{self.path}/preload_{self.transforms[-1].height}_{self.transforms[-1].width}"

        self.preload = preload
        self.preloaded = False
        if self.preload:
            self.save_processed_()
            self.preloaded = True

    def save_processed_(self):

        os.makedirs(self.dir_name, exist_ok=True)
        self.dir_name += f"/{self.datatype}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            for i, e in enumerate(self.img_ids):
                # print(i, e)
                img, mask = self.__getitem__(i)
                np.save(f"{self.dir_name}/{e}_mask.npy", mask)
                np.save(f"{self.dir_name}/{e}_img.npy", img)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if self.preloaded:
            img = np.load(f"{self.dir_name}/{image_name}_img.npy")
            mask = np.load(f"{self.dir_name}/{image_name}_mask.npy")

        else:
            mask = make_mask(self.df, image_name)
            image_path = os.path.join(self.data_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
                mask = preprocessed['mask']

        return img, mask

    def __len__(self):
        return len(self.img_ids)


def prepare_loaders(path: str = '', bs: int = 4, num_workers: int = 0, preprocessing_fn=None):

    train = pd.read_csv(f'{path}/train.csv')
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[~train['EncodedPixels'].isnull(), 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts(). \
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42,
                                            stratify=id_mask_count['count'], test_size=0.1)

    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    train_dataset = CloudDataset(path=path, df=train, datatype='train', img_ids=train_ids,
                                 transforms=get_training_augmentation1(),
                                 preprocessing=get_preprocessing(preprocessing_fn), preload=True)
    valid_dataset = CloudDataset(path=path, df=train, datatype='valid', img_ids=valid_ids,
                                 transforms=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn), preload=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    test_dataset = CloudDataset(path=path, df=sub, datatype='test', img_ids=test_ids,
                                transforms=get_validation_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn), preload=True)
    test_loader = DataLoader(test_dataset, batch_size=bs // 2, shuffle=False, num_workers=0)

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    return loaders
