# Mathematical
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image
# Pytorch
#import pandas as pd

import glob

from xml.etree import ElementTree as et
import torch
from torch.utils import data
from torchvision import datasets

# Misc
from functools import lru_cache
import cv2


def genuv(h, w):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    return np.stack([u, v], axis=-1)


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], axis=-1)


def xyz2uv(xyz):
    c = np.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = np.arctan2(xyz[..., 1], xyz[..., 0])
    v = np.arctan2(xyz[..., 2], c)
    return np.stack([u, v], axis=-1)


def uv2img_idx(uv, h, w, u_fov, v_fov, v_c=0):
    assert 0 < u_fov and u_fov < np.pi
    assert 0 < v_fov and v_fov < np.pi
    assert -np.pi < v_c and v_c < np.pi

    xyz = uv2xyz(uv.astype(np.float64))
    Ry = np.array([
        [np.cos(v_c), 0, -np.sin(v_c)],
        [0, 1, 0],
        [np.sin(v_c), 0, np.cos(v_c)],
    ])
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(v_c) * xyz[..., 0] - np.sin(v_c) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(v_c) * xyz[..., 0] + np.cos(v_c) * xyz[..., 2]
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) |\
              (v < -v_fov / 2) | (v > v_fov / 2)
    x[invalid] = -100
    y[invalid] = -100

    return np.stack([y, x], axis=0)

class OmniDataset(data.Dataset):
    def __init__(self, dataset, fov=120, outshape=(256, 256),
                 flip=False, h_rotate=False, v_rotate=False,
                 img_mean=None, img_std=None, fix_aug=False):
        '''
        Convert classification dataset to omnidirectional version
        @dataset  dataset with same interface as torch.utils.data.Dataset
                  yield (PIL image, label) if indexing
        '''
        self.dataset = dataset
        self.fov = fov
        self.outshape = outshape
        self.flip = flip
        self.h_rotate = h_rotate
        self.v_rotate = v_rotate
        self.img_mean = img_mean
        self.img_std = img_std

        self.aug = None
        if fix_aug:
            self.aug = [
                {
                    'flip': np.random.randint(2) == 0,
                    'h_rotate': np.random.randint(outshape[1]),
                    'v_rotate': np.random.uniform(-np.pi/2, np.pi/2),
                }
                for _ in range(len(self.dataset))
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = np.array(self.dataset[idx][0], np.float32)#/255.
        h, w = img.shape[:2]
        uv = genuv(*self.outshape)
        fov = self.fov * np.pi / 180

        if self.v_rotate:
            if self.aug is not None:
                v_c = self.aug[idx]['v_rotate']
            else:
                v_c = np.random.uniform(-np.pi/2, np.pi/2)
            img_idx = uv2img_idx(uv, h, w, fov, fov, v_c)
        else:
            img_idx = uv2img_idx(uv, h, w, fov, fov, 0)
        x = map_coordinates(img, img_idx, order=1)

        # Random flip
        if self.aug is not None:
            if self.aug[idx]['flip']:
                x = np.flip(x, axis=1)
        elif self.flip and np.random.randint(2) == 0:
            x = np.flip(x, axis=1)

        # Random horizontal rotate
        if self.h_rotate:
            if self.aug is not None:
                dx = self.aug[idx]['h_rotate']
            else:
                dx = np.random.randint(x.shape[1])
            x = np.roll(x, dx, axis=1)

        # Normalize image
        if self.img_mean is not None:
            x = x - self.img_mean
        if self.img_std is not None:
            x = x / self.img_std

        return torch.FloatTensor(x.copy()), self.dataset[idx][1]


class OmniDataset2(data.Dataset):
    def __init__(self, dataset, fov=120, outshape=(256, 256),
                 flip=False, h_rotate=False, v_rotate=False,
                 img_mean=None, img_std=None, fix_aug=False):
        '''
        Convert classification dataset to omnidirectional version
        @dataset  dataset with same interface as torch.utils.data.Dataset
                  yield (PIL image, label) if indexing
        '''
        
        #img = Image.open('cat0.jpg').convert('L')
        #img, label = dataset[0]
        #img.show()
        #rint(type(img))
        self.dataset = dataset
        self.dataset = img
        self.fov = fov
        self.outshape = outshape
        self.flip = flip
        self.h_rotate = h_rotate
        self.v_rotate = v_rotate
        self.img_mean = img_mean
        self.img_std = img_std

        self.aug = None
        if fix_aug:
            self.aug = [
                {
                    'flip': np.random.randint(2) == 0,
                    'h_rotate': np.random.randint(outshape[1]),
                    'v_rotate': np.random.uniform(-np.pi/2, np.pi/2),
                }
                for _ in range(len(self.dataset))
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = np.array(self.dataset, np.float32)
        h, w = img.shape[:2]
        uv = genuv(*self.outshape)
        fov = self.fov * np.pi / 180
        #print(img.shape)

        if self.v_rotate:
            if self.aug is not None:
                v_c = self.aug[idx]['v_rotate']
            else:
                v_c = np.random.uniform(-np.pi/2, np.pi/2)
            img_idx = uv2img_idx(uv, h, w, fov, fov, v_c)
        else:
            img_idx = uv2img_idx(uv, h, w, fov, fov, 0)
            #print(img_idx.shape)
        x = map_coordinates(img, img_idx, order=1)

        # Random flip
        if self.aug is not None:
            if self.aug[idx]['flip']:
                x = np.flip(x, axis=1)
        elif self.flip and np.random.randint(2) == 0:
            x = np.flip(x, axis=1)

        # Random horizontal rotate
        if self.h_rotate:
            if self.aug is not None:
                dx = self.aug[idx]['h_rotate']
            else:
                dx = np.random.randint(x.shape[1])
            x = np.roll(x, dx, axis=1)

        # Normalize image
        if self.img_mean is not None:
            x = x - self.img_mean
        if self.img_std is not None:
            x = x / self.img_std
        
        cv2.imwrite('cat.jpg', img) 

        return torch.FloatTensor(x.copy()), self.dataset

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        #self.idx = idx
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cat', 'dog']

    def __len__(self):
        return len(glob.glob(f'{self.root_dir}/*.jpg'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
        #                        self.landmarks_frame.iloc[idx, 0])
        img_name = f'{self.root_dir}/img_{idx}.jpg'
        #image = io.imread(img_name)
        image  = Image.open(img_name).convert('L')

        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}
        #sample = {'image': image}

        #if self.transform:
        #    sample = self.transform(sample)
                # capture the corresponding XML file for getting the annotations
        annot_filename = img_name[:-4] + '.xml'
        #annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_filename)
        root = tree.getroot()
        
        # get the height and width of the image
        image_width, image_height = image.size
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(float(member.find('bndbox').find('xmin').text))
            # xmax = right corner x-coordinates
            xmax = int(float(member.find('bndbox').find('xmax').text))
            # ymin = left corner y-coordinates
            ymin = int(float(member.find('bndbox').find('ymin').text))
            # ymax = right corner y-coordinates
            ymax = int(float(member.find('bndbox').find('ymax').text))
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = xmin#/image_width)#*self.image_width
            xmax_final = xmax#/image_width)#*self.image_width
            ymin_final = ymin#/image_height)#*self.image_height
            yamx_final = ymax#/image_height)#*self.image_height
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        
        #print(labels)
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.float32)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        return image, target["labels"]

class OmniCustom(OmniDataset):
    def __init__(self, root='/home/msnuel/trab-final-cv/images', train=True,
                 download=True, *args, **kwargs):
        
        self.custom = CustomDataset(root_dir = root)
        super(OmniCustom, self).__init__(self.custom, *args, **kwargs)


class OmniMNIST(OmniDataset):
    def __init__(self, root='datas/MNIST', train=True,
                 download=True, *args, **kwargs):
        '''
        Omnidirectional MNIST
        @root (str)       root directory storing the dataset
        @train (bool)     train or test split
        @download (bool)  whether to download if data now exist
        '''
        self.MNIST = datasets.MNIST(root, train=train, download=download)
        super(OmniMNIST, self).__init__(self.MNIST, *args, **kwargs)


class OmniFashionMNIST(OmniDataset):
    def __init__(self, root='datas/FashionMNIST', train=True,
                 download=True, *args, **kwargs):
        '''
        Omnidirectional FashionMNIST
        @root (str)       root directory storing the dataset
        @train (bool)     train or test split
        @download (bool)  whether to download if data now exist
        '''
        self.FashionMNIST = datasets.FashionMNIST(root, train=train, download=download)
        super(OmniFashionMNIST, self).__init__(self.FashionMNIST, *args, **kwargs)


if __name__ == '__main__':

    import os
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--idx', nargs='+', required=True,
                        help='image indices to demo')
    parser.add_argument('--out_dir', default='datas/demo',
                        help='directory to output demo image')
    parser.add_argument('--dataset', default='OmniCustom',
                        choices=['OmniMNIST', 'OmniFashionMNIST', 'OmniCustom'],
                        help='which dataset to use')

    parser.add_argument('--fov', type=int, default=120,
                        help='fov of the tangent plane')
    parser.add_argument('--flip', action='store_true',
                        help='whether to apply random flip')
    parser.add_argument('--h_rotate', action='store_true',
                        help='whether to apply random panorama horizontal rotation')
    parser.add_argument('--v_rotate', action='store_true',
                        help='whether to apply random panorama vertical rotation')
    parser.add_argument('--fix_aug', action='store_true',
                        help='whether to apply random panorama vertical rotation')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == 'OmniMNIST':
        dataset = OmniMNIST(fov=args.fov, flip=args.flip,
                            h_rotate=args.h_rotate, v_rotate=args.v_rotate,
                            fix_aug=args.fix_aug)
    if args.dataset == 'OmniCustom':
        dataset = OmniCustom(fov=args.fov, flip=args.flip,
                            h_rotate=args.h_rotate, v_rotate=args.v_rotate,
                            fix_aug=args.fix_aug)
    elif args.dataset == 'OmniFashionMNIST':
        dataset = OmniFashionMNIST(fov=args.fov, flip=args.flip,
                                   h_rotate=args.h_rotate, v_rotate=args.v_rotate,
                                   fix_aug=args.fix_aug)

    for idx in args.idx:
        idx = int(idx)
        path = os.path.join(args.out_dir, '%d.png' % idx)
        x, label = dataset[idx]

        print(path)
        Image.fromarray(x.numpy().astype(np.uint8)).save(path)
