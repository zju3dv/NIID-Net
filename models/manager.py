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
import shutil
from itertools import chain

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from models.niid_net import NIIDNet


class TrainState(object):
    isTraining = None  # current training state
    optim_NEM_coarse = None
    optim_NEM_refine = None
    optim_IID = None
    optimizer = None
    scheduler = None
    sd_type = None


class NIIDNetManager(object):
    net_name = None
    model = None

    train_state = TrainState()
    # cnt = 0
    gpu_devices = None
    data_gpu = None

    def __init__(self, opt):
        # Network Name
        self.net_name = 'NIID-Net'

        # Define Model
        self.model = NIIDNet()

        # GPU
        self.gpu_devices = opt.gpu_devices
        if self.gpu_devices is None:
            if opt.isTrain:
                raise Exception('Training code does not have CPU version.')
            self.data_gpu = None
            print('\nCPU version')
        else:
            print('\nGPU_devices: %s' % self.gpu_devices)
            self.data_gpu = self.gpu_devices[0]
            self.model = torch.nn.DataParallel(self.model.cuda(self.data_gpu), device_ids=self.gpu_devices)

        # Load pre-trained model and set optimizer
        self.reset_train_mode(opt)

    def reset_train_mode(self, opt):
        """ Load pre-trained model parameters and set optimizer and scheduler
        """
        self.train_state = TrainState()

        # Load Pretrained
        if opt.pretrained_file is not None:
            self._load_models(opt.pretrained_file,
                              load_NEM=opt.load_pretrained_NEM,
                              load_IID_Net=opt.load_pretrained_IID_Net)

        # Set optimizer and scheduler
        if opt.isTrain:
            self._set_optimizers_schedulers(opt)
            self.switch_to_train()
        else:
            self.switch_to_eval()

    def _get_framework_components(self):
        if self.gpu_devices is not None:
            return self.model.module.NEM_coarse_model, self.model.module.NEM_refine_model, self.model.module.IID_model
        else:
            return self.model.NEM_coarse_model, self.model.NEM_refine_model, self.model.IID_model

    def _load_models(self, file_path, load_NEM, load_IID_Net):
        print('\nLoading models from: %s' % file_path)

        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage) # load in CPU memory
        else:
            print("=> no checkpoint found at '{}'. Loading failed!".format(file_path))
            return

        NEM_coarse_model, NEM_refine_model, IID_Net = self._get_framework_components()

        # Load NEM
        if load_NEM:
            # NEM coarse
            d = checkpoint.get('NEM_coarse_state_dict')
            if d is not None:
                # delete_keys = ['stats_mean', 'stats_std']
                # for k in delete_keys:
                #     d.pop(k, None)
                # missing = set(self.GM_coarse_model.state_dict().keys()) - set(d.keys()) - set(delete_keys)
                # if len(missing) > 0:
                #     raise KeyError('missing keys in state_dict: "{}"'.format(missing))
                NEM_coarse_model.load_state_dict(d)
                print('    load NEM_coarse_model')
            else:
                print('    => load NEM_coarse_model failed, no state_dict in checkpoint file!')
            # NEM refine
            d = checkpoint.get('NEM_refine_state_dict')
            if d is not None:
                NEM_refine_model.load_state_dict(d)
                print('    load NEM_refine_model')
            else:
                print('    => load NEM_refine_model failed, no state_dict in checkpoint file!')

        # Load IID-Net
        if load_IID_Net:
            d = checkpoint.get('IID_state_dict')
            if d is not None:
                IID_Net.load_state_dict(d)
                print('    load IID-Net_model')
            else:
                print('    => load IID-Net_model failed,  no state_dict in checkpoint file!')

    def name(self):
        return self.net_name

    def switch_to_train(self, flag=True):  # return old training state
        if not flag:
            return self.switch_to_eval()

        old_isTrain = self.train_state.isTraining
        if (old_isTrain is None) or (not old_isTrain):
            NEM_coarse_model, NEM_refine_model, IID_Net = self._get_framework_components()
            NEM_coarse_model.train(self.train_state.optim_NEM_coarse)
            NEM_refine_model.train(self.train_state.optim_NEM_refine)
            if self.train_state.optim_IID is not None:
                if self.train_state.optim_IID == 'full':
                    IID_Net.train(True)
                elif self.train_state.optim_IID == 'wo_R':
                    IID_Net.encoder.train(True)
                    IID_Net.decoder_L.train(True)
                    IID_Net.decoder_R.train(False)
                elif self.train_state.optim_IID == 'R':
                    IID_Net.encoder.train(False)
                    IID_Net.decoder_L.train(False)
                    IID_Net.decoder_R.train(True)
                else:
                    raise Exception('Undefined optim_IID type: %s' % self.train_state.optim_IID)
            else:
                IID_Net.train(False)
            self.train_state.isTraining = True

        return {'flag': old_isTrain}

    def switch_to_eval(self):  # return old training state
        old_isTrain = self.train_state.isTraining
        if (old_isTrain is None) or (old_isTrain):
            self.model.eval()
            self.train_state.isTraining = False
        return {'flag': old_isTrain}

    def save(self, path, label, epoch=None, nyu_val=None, iiw_val=None, saw_sub_val=None,
             best_normal=False, best_IID=False):
        NEM_coarse_model, NEM_refine_model, IID_Net = self._get_framework_components()
        checkpoint = {
            'epoch': epoch,

            'NEM_coarse_state_dict': NEM_coarse_model.state_dict(),
            'NEM_refine_state_dict': NEM_refine_model.state_dict(),
            'IID_state_dict': IID_Net.state_dict(),

            'nyu_val': nyu_val,
            'iiw_val': iiw_val,
            'saw_sub_val': saw_sub_val,
            # 'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
        }
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, label+'_.pth.tar')
        torch.save(checkpoint, filepath)
        print('Save checkpoint file: %s' % filepath)
        if best_normal:
            shutil.copyfile(filepath, os.path.join(path, 'best_normal_model.pth.tar'))
        if best_IID:
            shutil.copyfile(filepath, os.path.join(path, 'best_IID_model.pth.tar'))
        return filepath

    def _define_lr_scheduler(self, optimizer, opt):
        if opt.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau \
                (optimizer, mode=opt.scheduler_mode, factor=opt.lr_decay, patience=opt.sd_patience, min_lr=opt.min_lr, verbose=True)
            print('    ReduceLROnPlateau: lr_decay:%.6f, sd_patience:%.6f, min_lr:%.6f' %
                  (opt.lr_decay, opt.sd_patience, opt.min_lr))
        elif opt.scheduler_type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_gamma)
            print('    ExponentialLR: lr_gamma:%.6f' % (opt.lr_gamma,))
        elif opt.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.lr_t_max, eta_min=opt.min_lr)
            print('    CosineAnnealingLR: lr_t_max:%.6f, min_lr:%.6f' % (opt.lr_t_max, opt.min_lr))
        else:
            raise Exception('Error: not support lr_scheduler type: %s' % opt.scheduler_type)
        return scheduler, opt.scheduler_type

    def _set_optimizers_schedulers(self, opt):
        # Get parameters
        print('\nGetting parameters to be optimized:')

        NEM_coarse_model, NEM_refine_model, IID_Net = self._get_framework_components()
        learning_rate = opt.lr
        optim_params = []

        if opt.optim_NEM_coarse:
            optim_params.append({'params': NEM_coarse_model.parameters(),
                                 'lr': learning_rate,
                                 'name': 'NEM_coarse_net'})
            print('    parameters from NEM_coarse_model, lr %f' % learning_rate)
        for param in NEM_coarse_model.parameters():
            param.requires_grad = opt.optim_NEM_coarse

        if opt.optim_NEM_refine:
            optim_params.append({'params': NEM_refine_model.parameters(),
                                 'lr': learning_rate,
                                 'name': 'NEM_refine_net'})
            print('    parameters from NEM_refine_model, lr %f' % learning_rate)
        for param in NEM_refine_model.parameters():
            param.requires_grad = opt.optim_NEM_refine

        for param in IID_Net.parameters():
            param.requires_grad = opt.optim_IID is not None
        if opt.optim_IID is not None:
            if opt.optim_IID == 'full':
                optim_params.append({'params': IID_Net.parameters(),
                                     'lr': learning_rate,
                                     'name': 'IID-Net'})
                print('    parameters from IID-Net, lr %f' % learning_rate)
            elif opt.optim_IID == 'wo_R':
                optim_params.append({'params': chain(IID_Net.encoder.parameters(),
                                                     IID_Net.decoder_L.parameters()),
                                     'lr': learning_rate,
                                     'name': 'IID-Net_S_branch'})
                for param in IID_Net.decoder_R.parameters():
                    param.requires_grad = False
                print('    parameters from IID-Net_S_branch, lr %f' % learning_rate)
            elif opt.optim_IID == 'R':
                optim_params.append({'params': IID_Net.decoder_R.parameters(),
                                     'lr': learning_rate,
                                     'name': 'IID-Net_R_decoder'})
                for param in IID_Net.encoder.parameters():
                    param.requires_grad = False
                for param in IID_Net.decoder_L.parameters():
                    param.requires_grad = False
                print('    parameters from IID-Net_R_decoder, lr %f' % learning_rate)
            else:
                raise Exception('Undefined optim_IID type: %s' % opt.optim_IID)


        # Optimizer and scheduler
        option = opt
        if option.use_SGD:
            optimizer = torch.optim.SGD(optim_params, lr=option.lr, momentum=0.9, weight_decay=option.weight_decay)
        else:
            optimizer = torch.optim.Adam(optim_params, lr=option.lr, betas=(0.9, 0.999),
                                         weight_decay=option.weight_decay)
        # print('    lr:%.6f, weight_decay:%.6f' % (option.lr, option.weight_decay))
        scheduler, sd_type = self._define_lr_scheduler(optimizer, option)

        self.train_state.optim_NEM_coarse = opt.optim_NEM_coarse
        self.train_state.optim_NEM_refine = opt.optim_NEM_refine
        self.train_state.optim_IID = opt.optim_IID
        self.train_state.optimizer = optimizer
        self.train_state.scheduler = scheduler
        self.train_state.sd_type = sd_type

    def _forward(self, input_srgb, pred_normal, pred_reflect, pred_shading):
        out_N, out_R, out_L, out_S, rendered_img = self.model(input_srgb, pred_normal, pred_reflect, pred_shading)
        return out_N, out_R, out_L, out_S, rendered_img

    def predict(self, inputs, normal=False, IID=False):
        # switch to eval mode
        self.switch_to_eval()

        # Input Data
        input_srgb = inputs['input_srgb'].float()
        if self.gpu_devices is not None:
            input_srgb = input_srgb.cuda(self.data_gpu)
        input_srgb = Variable(input_srgb, volatile=True)

        # Forward
        N, R, L, S, rendered_img = self._forward(input_srgb,
                                                 pred_normal=normal,
                                                 pred_reflect=IID,
                                                 pred_shading=IID)

        if N is not None:
            N = N / torch.norm(N, p=2, dim=1, keepdim=True).clamp(min=1e-6)
            N = N.data
        if R is not None:
            R = R.data
        if L is not None:
            L = L.data
        if S is not None:
            S = (S.repeat(1, 3, 1, 1)).data
        if rendered_img is not None:
            rendered_img = rendered_img.data

        return N, R, L, S, rendered_img

    def predict_IID_np_for_saw_eval(self, saw_img):
        # Input Data
        saw_img = np.transpose(saw_img, (2, 0, 1))
        input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()

        p_N, p_R, p_L, p_S, rendered_img = self.predict({'input_srgb': input_}, IID=True)

        p_N_np = np.transpose(p_N[0, :, :, :].cpu().numpy(), (1, 2, 0))
        p_L_np = np.transpose(p_L[0, :, :, :].cpu().numpy(), (1, 2, 0))
        p_S_np = np.transpose(p_S[0, :, :, :].cpu().numpy(), (1, 2, 0))
        p_R_np = np.transpose(p_R[0, :, :, :].cpu().numpy(), (1, 2, 0))
        rendered_img_np = np.transpose(rendered_img[0, :, :, :].cpu().numpy(), (1, 2, 0))

        return p_N_np, p_R_np, p_L_np, p_S_np, rendered_img_np

    def set_evaluation_results(self, eval):
        scheduler = self.train_state.scheduler
        sd_type = self.train_state.sd_type
        if sd_type == 'plateau':
            scheduler.step(metrics=eval, epoch=None)
        else:
            scheduler.step(epoch=None)


def create_model(opt):
    model = NIIDNetManager(opt)
    print("\nModel [%s] is created!\n" % (model.name()))
    return model

