# encoding=utf8

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
import warnings
import torch


class CriteriaTypes(object):
    normal = 0
    warm_up = 1
    shading_warm_up = 2
    shading = 3
    reflectance = 4
    IID = 5

    @classmethod
    def is_valid(cls, label):
        return label in [cls.normal, cls.warm_up, cls.shading_warm_up, cls.shading, cls.reflectance, cls.IID]

    @classmethod
    def train_shading(cls, label):
        return label in [cls.warm_up, cls.shading_warm_up, cls.shading, cls.IID]

    @classmethod
    def train_reflectance(cls, label):
        return label in [cls.warm_up, cls.reflectance, cls.IID]

    @classmethod
    def train_surface_normal(cls, label):
        return label in [cls.normal]

    @classmethod
    def warm_up_shading(cls, label):
        return label in [cls.warm_up, cls.shading_warm_up]


class DefaultConfig(object):
    # Checkpoints dir (default)
    checkpoints_dir = './checkpoints/'

    # Visdom
    env = 'default'
    server = 'http://localhost'
    port = 18097
    offline = True

    # Dataset Path Config
    dataset_root = './dataset/'
    DIODE_root = os.path.join(dataset_root, 'DIODE/')
    NYUv2_root = os.path.join(dataset_root, 'nyu_depth_v2_large/')
    CGI_root = os.path.join(dataset_root, 'CGIntrinsics/')
    diode_meta_fname = os.path.join(DIODE_root, 'diode_meta.json')

    # Input
    normal_input_size = (240, 320)
    batch_size_normal = 1
    num_workers_normal = 1
    batch_size_intrinsics = 1
    batch_size_iiw = 1
    batch_size_saw = 1
    num_workers_intrinsics = 1
    num_workers_Render = 1

    # Model
    model = 'NIID-Net'
    pretrained_file = './pretrained_model/final.pth.tar'
    load_pretrained_NEM = False
    load_pretrained_IID_Net = False

    # State
    isTrain = False

    # GPU
    use_gpu = True
    gpu_devices = [0]

    # Parse
    def parse(self, kwargs):
        """ Update configuration according to kwargs
        """
        print('\nUpdate config:')
        for k, v in kwargs.items():
            # GPU NUM
            if k == 'gpu_num':
                self.gpu_devices = [n + torch.cuda.current_device() for n in range(int(v))]
                print('    gpu_devices: %s' % self.gpu_devices)
                continue

            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
            print('    %s: %s' % (k, v))

        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('__'):
        #         print(k, getattr(self, k))
        print('\n')


class TestOptions(DefaultConfig):
    # State
    isTrain = False

    # Pretrained
    load_pretrained_NEM = True
    load_pretrained_IID_Net = True

    # Visualization
    env = 'NIID-Net_test'  # visdom environment


class TrainIIDOptions(DefaultConfig):
    # Pretrained
    load_pretrained_NEM = True
    load_pretrained_IID_Net = False

    # Train
    isTrain = True
    train_epoch = 50
    iter_each_epoch = None
    criteria = None

    # Optimizer
    use_SGD = False
    lr = 1e-4  # default learning rate
    weight_decay = 0e-4
    optim_NEM = False
    optim_IID = 'wo_R'  # None, 'full', 'wo_R' (train without the R decoder), 'R' (only R decoder)

    # Scheduler
    scheduler_type = 'plateau'
    scheduler_mode = 'max'
    lr_decay = 0.5
    sd_patience = 5
    min_lr = 1e-6

    # Visualization
    env = 'NIID-Net_train_IIDNet'


class TrainNormalOptions(DefaultConfig):
    # Pretrained
    load_pretrained_NEM = True
    load_pretrained_IID_Net = False

    # Train
    isTrain = True
    train_epoch = 50
    iter_each_epoch = None
    criteria = CriteriaTypes.normal

    # Optimizer
    use_SGD = False
    lr = 5e-5  # default learning rate
    weight_decay = 1e-4
    optim_NEM = True
    optim_IID = None

    # Scheduler
    scheduler_type = 'plateau'
    scheduler_mode = 'max'
    lr_decay = 0.5
    sd_patience = 5
    min_lr = 1e-8

    # Visualization
    env = 'NIID-Net_train_NEM'
