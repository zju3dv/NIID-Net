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
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image

from utils.visualize import Visualizer
vis = Visualizer('vis_nyu_data_loader')

# Data directory
nyu_dir = 'dataset/nyu_depth_v2/'
splits_path = nyu_dir + 'splits.mat'
input_rgb_images_dir = nyu_dir + 'nyu_datasets_changed/input/'
target_depth_images_dir = nyu_dir + 'nyu_datasets_changed/target_depths/'
#target_labels_images_dir = 'data/nyu_datasets_changed/labels_38/'

# Batch
## Whether to use a subset of the dataset to train/test
use_subset = False
train_subset = 16
val_subset = 16
test_subset = 16


def srgb_to_rgb(srgb):
    ret = torch.zeros(srgb.size()).float()
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


class NYUDataset(data_utils.Dataset):
    def __init__(self, data_dir, listing, target_size, train=True):
        self.data_dir = data_dir
        self.listing = listing
        self.input_target_size, self.depth_target_size = target_size
        self.train = train

        # Pre transform
        # self.pre_co_transform = transforms.Compose([
        #     # transforms.CenterCrop((460, 620)),
        #     transforms.Resize(290, interpolation=Image.BILINEAR),
        # ])
        self.input_pre_transform = transforms.Resize(290, interpolation=Image.BILINEAR)
        self.depth_pre_transform = transforms.Resize(290, interpolation=Image.NEAREST)

        # Data Augmentation
        if self.train:
            self.da_co_transforms = transforms.Compose((
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=5.0, resample=Image.BILINEAR)
            ))
            self.da_input = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4)
            ])

        # Post transform
        self.post_co_transform = transforms.Compose([
            transforms.CenterCrop(self.input_target_size)
        ])
        self.input_post_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_post_transform = transforms.Compose([
            transforms.Resize(self.depth_target_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # Data path
        input_dir,target_depth_dir = self.data_dir
        file_prefix=self.listing[index]
        img_name = file_prefix+'.jpg'
        target_depth_name = file_prefix+'.npy'

        # Read data
        input_im = Image.open(os.path.join(input_dir, img_name))
        target_depth_np = np.load(os.path.join(target_depth_dir, target_depth_name))
        target_depth_im = transforms.ToPILImage()(np.expand_dims(target_depth_np, axis=2))

        # Pre Transform
        # input_im, target_depth_im = self.pre_co_transform(input_im, target_depth_im)
        input_im = self.input_pre_transform(input_im)
        target_depth_im = self.depth_pre_transform(target_depth_im)

        # Data Augmentation
        if self.train:
            input_im, target_depth_im = self.da_co_transforms(input_im, target_depth_im)
            input_im = self.da_input(input_im)

        # Post Transform
        input_im, target_depth_im = self.post_co_transform(input_im, target_depth_im)
        raw_im = transforms.ToTensor()(input_im)
        input_im = self.input_post_transform(input_im)
        target_depth_im_large = transforms.ToTensor()(target_depth_im)
        target_depth_im = self.depth_post_transform(target_depth_im)

        rgb_img = srgb_to_rgb(input_im)
        mask_depth = (target_depth_im > 1e-6).float()

        inputs = {'raw_srgb': raw_im,
                  'input_srgb': input_im}
        targets = {'depth': target_depth_im,
                   'depth_large': target_depth_im_large,
                   'rgb_img': rgb_img,
                   'mask_depth': mask_depth,
                   'index': index}

        # vis.img('raw_img', raw_im)
        # vis.img('input_srgb', input_im)
        # vis.img('rgb_img', rgb_img)

        return {'inputs': inputs, 'targets': targets}

    def __len__(self):
        return len(self.listing)


def get_loader(target_input_size=(228, 304), target_depth_size=(114, 152), batch_size=16, num_workers=2, split_val=True):
    print('\nLoading images...')

    # Split dataset into train, validation and test
    splits_dir = splits_path
    splits = sio.loadmat(splits_dir)
    train_val_split = splits['trainNdxs'] - 1  # indexes start at 1 in splits.mat
    test_split = splits['testNdxs'] - 1
    if split_val:
        NUM_TRAIN = int(len(train_val_split) * 0.80)
    else:
        NUM_TRAIN = len(train_val_split)
    NUM_VAL = len(train_val_split) - NUM_TRAIN
    NUM_TEST = len(test_split)

    ## Whether to use a subset of the dataset
    train_on = NUM_TRAIN
    val_on = NUM_VAL
    test_on = NUM_TEST
    if use_subset:
        train_on = train_subset
        val_on = val_subset
        test_on = test_subset

    # Load Data
    assert len(os.listdir(input_rgb_images_dir)) == NUM_TRAIN+NUM_VAL+NUM_TEST, "incomplete dataset"  ## Validate if the dataset is complete
    ## train+validate
    # Shuffle imgs
    # np.random.seed(100)
    # train_val_split = np.random.permutation(train_val_split)
    train_listing = train_val_split[:min(NUM_TRAIN,train_on)]
    train_listing = ['%05d'%item for sublist in train_listing for item in sublist]
    val_listing = train_val_split[NUM_TRAIN:min(NUM_VAL+NUM_TRAIN,val_on+NUM_TRAIN)]
    val_listing = ['%05d'%item for sublist in val_listing for item in sublist]
    ## test
    test_listing = test_split[:min(NUM_TEST, test_on)]
    test_listing = ['%05d'%item for sublist in test_listing for item in sublist]
    print('Split dataset: ')
    print('  size of train: %d' % len(train_listing))
    print('  size of val: %d' % len(val_listing))
    print('  size of test: %d' % len(test_listing))

    ## Dataset
    ## Splitting in train, val and test sets [No data augmentation on val and test, only on train]
    data_dir = (input_rgb_images_dir, target_depth_images_dir)
    target_size = (target_input_size, target_depth_size)
    train_dataset = NYUDataset(data_dir, train_listing, target_size, train=True)
    val_dataset = NYUDataset(data_dir, val_listing, target_size, train=False)
    test_dataset = NYUDataset(data_dir, test_listing, target_size, train=False)

    ## Data Loader
    print("Loading data...")
    train_loader = data_utils.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    print("Loaded data\n")
    return train_loader, val_loader, test_loader
