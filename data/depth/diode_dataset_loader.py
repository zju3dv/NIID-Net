# ////////////////////////////////////////////////////////////////////////////
# //  This file is part of NIID-Net. For more information
# //  see <https://github.com/zju3dv/NIID-Net>.
# //  If you use this code, please cite the corresponding publications as
# //  listed on the above website.
# //
# //  Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.
# //
# //  Permission to use, copy, modify and distribute this software and its
# //  documentation for educational, research and non-profit purposes only.
# //
# //  The above copyright notice and this permission notice shall be included in all
# //  copies or substantial portions of the Software.
# //
# //  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# // SOFTWARE.
# ////////////////////////////////////////////////////////////////////////////


import os.path as osp
from itertools import chain
import json

import torch
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import numpy as np
from PIL import Image
from skimage.transform import resize

import matplotlib.pyplot as plt

from utils import joint_transforms_tv021 as transforms

'''
The json metadata for DIODE is laid out as follows:
train:
    outdoor:
        scene_000xx:
            scan_00yyy:
                - 000xx_00yyy_indoors_300_010
                - 000xx_00yyy_indoors_300_020
                - 000xx_00yyy_indoors_300_030
        scene_000kk:
            _analogous_
val:
    _analogous_
test:
    _analogous_
'''

_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, )
    for split in tokens:
        assert split in valid_tokens
    return tokens


def enumerate_paths(src):
    '''flatten out a nested dictionary into an iterable
    DIODE metadata is a nested dictionary;
    One could easily query a particular scene and scan, but sequentially
    enumerating files in a nested dictionary is troublesome. This function
    recursively traces out and aggregates the leaves of a tree.
    '''
    if isinstance(src, list):
        return src
    elif isinstance(src, dict):
        acc = []
        for k, v in src.items():
            _sub_paths = enumerate_paths(v)
            _sub_paths = list(map(lambda x: osp.join(k, x), _sub_paths))
            acc.append(_sub_paths)
        return list(chain.from_iterable(acc))
    else:
        raise ValueError('do not accept data type {}'.format(type(src)))


def plot_depth_map(dm, validity_mask):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))


def plot_normal_map(normal_map):
    normal_viz = normal_map[:, ::, :]

    normal_viz = normal_viz + np.equal(np.sum(normal_viz, 2, 
    keepdims=True), 0.).astype(np.float32)*np.min(normal_viz)

    normal_viz = (normal_viz - np.min(normal_viz))/2.
    plt.axis('off')
    plt.imshow(normal_viz)


class DIODE_Dataset(Dataset):
    eps = 1e-6
    bit_8_img = 255.0

    def __init__(self, meta_fname, data_root, splits, scene_types, target_size):
        self.data_root = data_root
        self.splits = check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )

        # train list
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)
        imgs = []
        for split in self.splits:
            for scene_type in self.scene_types:
                _curr = enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: osp.join(split, scene_type, x), _curr)
                imgs.extend(list(_curr))
        self.imgs = imgs
        # lost data
        self.imgs = [item for item in self.imgs\
                     if not item.startswith('train/indoors/scene_00000/scan_00007')]
        self.imgs.sort()

        # transformation
        self.is_train = (splits == 'train')
        if self.is_train:
            self.da_input_img = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4)
            ])
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.target_size = target_size

    def __len__(self):
        return len(self.imgs)

    def normalize_surface_normal(self, normal, mask_normal):
        length = torch.sum(normal ** 2, dim=0, keepdim=True) ** 0.5
        mask_normal *= (length > self.eps).float()
        length[length < self.eps] = self.eps
        normal = (normal / length) * mask_normal
        return normal, mask_normal

    def __getitem__(self, index):
        im = self.imgs[index]
        im_fname = osp.join(self.data_root, '{}.png'.format(im))
        de_fname = osp.join(self.data_root, '{}_depth.npy'.format(im))
        de_mask_fname = osp.join(self.data_root, '{}_depth_mask.npy'.format(im))
        norm_fname = osp.join(self.data_root, '{}_normal.npy'.format(im))
        # Load data
        im = Image.open(im_fname)
        if self.is_train:
            im = self.da_input_img(im)
        im = np.array(im) / self.bit_8_img
        de = np.load(de_fname).squeeze()[:, :, np.newaxis]
        de_mask = np.load(de_mask_fname).squeeze()[:, :, np.newaxis]
        norm = -np.load(norm_fname).astype(np.float32)
        norm[:, :, 0] = -norm[:, :, 0]
        norm_mask = (np.sum(norm ** 2, axis=2) > self.eps).astype(np.float32).squeeze()[:, :, np.newaxis]

        # Transform
        data_tuple = (im, de, de_mask, norm, norm_mask)
        data_tuple = (resize(item, self.target_size, order=1, mode='constant', cval=0, preserve_range=True, anti_aliasing=True).astype(np.float32) for item in
                      data_tuple)
        data_tuple = self.transform(*data_tuple)
        input_im, depth_im, mask_depth, normal_im, mask_normal = data_tuple
        mask_depth = (mask_depth > 0.99).float()
        mask_normal = (mask_normal > 0.99).float()
        normal_im, mask_normal = self.normalize_surface_normal(normal_im, mask_normal)
        raw_srgb = input_im

        inputs = {'raw_srgb': raw_srgb,
                  'input_srgb': input_im}
        targets = {'depth': depth_im,
                   'mask_depth': mask_depth,
                   'normal': normal_im,
                   'valid_normal': mask_normal,
                   'index': index}

        return {'inputs': inputs, 'targets': targets, 'index': index}


def get_loader(meta_fname, data_root, is_train, is_indoor_scene, target_size, batch_size, num_workers):
    # only have train/val split
    split = 'train' if is_train else 'val'
    scene_type = 'indoors' if is_indoor_scene else 'outdoor'
    dset = DIODE_Dataset(meta_fname, data_root, splits=split, scene_types=scene_type, target_size=target_size)
    print('Size of DIODE %s_%s subset: %d' % (split, scene_type, len(dset)))
    data_loader = data_utils.DataLoader(
        dset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, drop_last=is_train
    )
    return data_loader
