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


import os
import os.path

import torch
import torch.utils.data as data_utils
from utils import joint_transforms_tv021 as transforms
import numpy as np
# matplotlib.use('Agg')
from PIL import Image
import pandas as pd
import h5py


def srgb_to_rgb(srgb):
    ret = torch.zeros(srgb.size()).float()
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


class NYUDataset(data_utils.Dataset):
    # Statistics
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    #
    bit_8_img = 255.0
    eps = 1e-6

    def __init__(self, data_dir, csv_file, target_size, train):
        self.data_dir = data_dir
        self.frame = pd.read_csv(csv_file, header=None)
        self.target_size = target_size
        self.train = train
        if not self.train:
            self.test_normal_dir = os.path.join(self.data_dir, 'nyu2_test_normal_GeoNet')
            self.test_normals = None
            self.test_normal_masks = None

        # Pre transform
        if self.train:
            self.pre_co_transform = transforms.Compose([
                transforms.RandomCrop((384, 512), padding=0, pad_if_needed=False),
                transforms.Resize(self.target_size, interpolation=Image.BILINEAR)
            ])
        else:
            self.pre_co_transform = transforms.Compose([
                # transforms.Resize(290, interpolation=Image.BILINEAR),
                # transforms.CenterCrop((426, 560)),
                # transforms.CenterCrop((384, 512)),
                transforms.Resize(self.target_size, interpolation=Image.BILINEAR)
            ])


        # Data Augmentation
        if self.train:
            # self.da_co_transforms = transforms.Compose((
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation(degrees=5.0, resample=Image.BILINEAR)
            # ))
            self.da_input = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4)
            ])

        # Post transform
        self.post_co_transform = transforms.Compose([
            # transforms.CenterCrop(self.target_size),
            transforms.ToTensor()
        ])
        self.to_tensor = transforms.ToTensor()
        # self.input_post_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=self.__imagenet_stats['mean'], std=self.__imagenet_stats['std'])
        # ])
        # self.depth_post_transform = transforms.Compose([
        #     transforms.Resize(self.depth_target_size, interpolation=Image.BILINEAR),
        #     # transforms.ToTensor()
        # ])

    def normalize_surface_normal(self, normal, valid_mask):
        if isinstance(normal, torch.Tensor):
            length = torch.sum(normal ** 2, dim=0, keepdim=True) ** 0.5
            valid_mask *= (length > self.eps).float()
        else:
            length = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
            valid_mask *= length > self.eps
        length[length < self.eps] = self.eps
        normal = (normal / length) * valid_mask
        return normal, valid_mask

    def __getitem__(self, index):
        # Data Path
        image_name = self.data_dir + self.frame.iloc[index, 0][5:]
        depth_name = self.data_dir + self.frame.iloc[index, 1][5:]
        ## pre-computed surface normal file path
        if not self.train:
            # name_prefix = self.frame.ix[index, 0][15: 20]
            # normal_name = self.data_dir + 'nyu2_test_normal/' + name_prefix + '_normal.npy'
            # valid_normal_name = self.data_dir + 'nyu2_test_normal/' + name_prefix + '_depthValid.npy'
            img_idx = int(self.frame.iloc[index, 0][15: 20])
        else:
            subdir, name_prefix = os.path.split(depth_name)
            subdir = subdir[subdir.find("nyu2_train/") + len("nyu2_train/"):]
            name_prefix = name_prefix[:-4]
            normal_name = os.path.join(self.data_dir, 'nyu2_train_normal/',  subdir, name_prefix + '_normal.png')

        # Load Data
        input_im = Image.open(image_name)
        target_depth_im = Image.open(depth_name)
        if not self.train:
            # normal_np = -np.load(normal_name).astype(np.float32)
            # normal_np = normal_np.transpose(1, 2, 0) # [H, W, C]
            # valid_normal_np = np.load(valid_normal_name).astype(np.float32)
            # valid_normal_np = np.expand_dims(valid_normal_np, axis=2) # [H, W, C]

            if self.test_normals is None or self.test_normal_masks is None:
                # Note the surface normals provided by GeoNet is in a left-handed coordinate system: x-left, y-up, z-from screen to viewer
                self.test_normals = h5py.File(os.path.join(self.test_normal_dir, 'norm_gt_l.mat'), 'r')['norm_gt_l']
                self.test_normal_masks = h5py.File(os.path.join(self.test_normal_dir, 'masks.mat'), 'r')['masks']
            normal_np = -self.test_normals[img_idx].transpose(2, 1, 0).astype(np.float32)  # [H, W, C]
            valid_normal_np = self.test_normal_masks[img_idx][:, :, np.newaxis].transpose(1, 0, 2).astype(np.float32)  # [H, W, C]

            # normal_np = normal_np[44:470, 40:600]
            # valid_normal_np = valid_normal_np[44:470, 40:600]
            # input_im = input_im.crop((40, 44, 600, 470))
            # target_depth_im = target_depth_im.crop((40, 44, 600, 470))
        else:
            normal_np = np.array(Image.open(normal_name)).astype(np.float32) / self.bit_8_img
            normal_np = -(normal_np * 2 - 1)
            valid_normal_np = np.zeros((normal_np.shape[0], normal_np.shape[1], 1)).astype(np.float32)
            valid_normal_np[44:470, 40:600] = 1.0
            # valid_normal_np = (np.sum(normal_np**2, axis=2) > self.eps).astype(np.float32)[..., np.newaxis]
        normal_np, valid_normal_np = self.normalize_surface_normal(normal_np, valid_normal_np)

        # Transform

        ## convert each dim to PILImage
        normal_np = (normal_np + 1) / 2.0  # change to positive temporarily
        normal_x_im = transforms.ToPILImage()(normal_np[:, :, 0:1])
        normal_y_im = transforms.ToPILImage()(normal_np[:, :, 1:2])
        normal_z_im = transforms.ToPILImage()(normal_np[:, :, 2:3])
        valid_normal_im = transforms.ToPILImage()(valid_normal_np)

        ## Data tuple
        data_set = (input_im, target_depth_im, normal_x_im, normal_y_im, normal_z_im, valid_normal_im)

        ## pre-transform
        data_set = (img.crop((40, 44, 600, 470)) for img in data_set)
        data_set = self.pre_co_transform(*data_set)
        ## Data Augmentation
        if self.train:
            # data_set = self.da_co_transforms(*data_set)
            tmp = self.da_input(data_set[0])
            data_set = (tmp, *data_set[1:])
        ## post-transform
        data_set = self.post_co_transform(*data_set)

        # To Tensor
        input_im = raw_im = data_set[0]
        rgb_img = srgb_to_rgb(input_im)
        target_depth_im = data_set[1]
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        if self.train:
            target_depth_im = target_depth_im.float() * 10.0
        else:
            target_depth_im = target_depth_im.float() / 1000.0
        mask_depth = (target_depth_im > 1e-6).float()

        ## surface normal and valid mask
        n_x_im, n_y_im, n_z_im, valid_normal_im = data_set[2:]
        normal_im = torch.cat((n_x_im, n_y_im, n_z_im), dim=0)
        normal_im = normal_im * 2 - 1.0 # recover
        valid_normal_im[valid_normal_im < 0.99] = 0.0 # unvalid
        valid_normal_im[valid_normal_im >= 0.99] = 1.0 # valid
        ## normalize
        normal_im, valid_normal_im = self.normalize_surface_normal(normal_im, valid_normal_im)

        # Visualizer.vis.img_many({
        #     # 'raw_srgb': raw_im,
        #     'input_srgb': input_im,
        #     'target_depth': target_depth_im / torch.max(target_depth_im),
        #     # 'rgb_img': rgb_img,
        #     # 'mask_depth': mask_depth,
        #     'normal': (-normal_im+1.0)/2.0,
        #     'valid_normal': valid_normal_im,
        # })

        inputs = {'raw_srgb': raw_im,
                  'input_srgb': input_im,}
        targets = {'depth': target_depth_im,
                   # 'depth_large': target_depth_im_large,
                   'mask_depth': mask_depth,
                   'normal': normal_im,
                   'valid_normal': valid_normal_im,
                   'rgb_img': rgb_img,
                   'index': index}

        return {'inputs': inputs, 'targets': targets, 'index': index}

    def __len__(self):
        return len(self.frame)


def get_loader(nyu_dir, is_train, target_size, batch_size, num_workers):

    csv_file = os.path.join(nyu_dir, 'nyu2_train.csv' if is_train else 'nyu2_test.csv')
    nyu_dataset = NYUDataset(nyu_dir, csv_file, target_size, train=is_train)
    nyu_loader = data_utils.DataLoader(
        nyu_dataset, batch_size, shuffle=is_train, num_workers=num_workers, drop_last=is_train, pin_memory=False
    )
    print('nyu_v2_large dataset:')
    print('    size of %s: %d' % ('train' if is_train else 'test', len(nyu_dataset)))
    return nyu_loader


def get_train_loader(batch_size, num_workers):
    # target_size = (target_input_size, target_depth_size)

    # train_data_loader
    train_dataset = NYUDataset(nyu_dir, nyu_dir + 'nyu2_train.csv', target_size, train=True)
    train_loader = data_utils.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                         drop_last=True, pin_memory=False)

    print('nyu_v2_large dataset:')
    print('    size of train: %d' % len(train_dataset))
    return train_loader


def get_test_loader(batch_size, num_workers):
    # target_size = (target_input_size, target_depth_size)

    # test_data_loader
    test_dataset = NYUDataset(nyu_dir, nyu_dir + 'nyu2_test.csv', target_size, train=False)
    test_loader = data_utils.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                        drop_last=False, pin_memory=False)
    print('nyu_v2_large dataset:')
    print('    size of test: %d' % len(test_dataset))
    return test_loader
