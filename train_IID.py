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


import time
import os

from config import TrainIIDOptions, CriteriaTypes
from data.intrinsics.data_loader import CreateDataLoaderCGIntrinsics
from data.intrinsics.data_loader import CreateDataLoaderRender
from models.manager import create_model
import utils.visualize as V
from evaluate import validate_iiw
from evaluate import validate_saw
from utils import pytorch_settings


tg_WHDR = 0.166
tg_AP = 0.984


def _train_CGIntrinsics(model, opt, print_label='CGIntrinsics'):
    # visualizer
    V.create_a_visualizer(opt)

    # data loader
    full_root = opt.dataset_root
    print('batch size: intrinsics %d' % opt.batch_size_intrinsics)

    train_list_CGIntrinsics = full_root + 'CGIntrinsics/intrinsics_final/train_list/'
    data_loader_S = CreateDataLoaderCGIntrinsics(full_root, train_list_CGIntrinsics,
                                                 _batch_size=opt.batch_size_intrinsics,
                                                 _num_workers=opt.num_workers_intrinsics)
    dataset_CGIntrinsics = data_loader_S.load_data()
    dataset_size_CGIntrinsics = len(data_loader_S)
    print("train_list CGIntrinsics Intrinsics = %d \n" % dataset_size_CGIntrinsics)

    train_list_Render = full_root + 'CGIntrinsics/intrinsics_final/render_list/'
    data_loader_Render = CreateDataLoaderRender(full_root, train_list_Render, _num_workers=opt.num_workers_Render)
    dataset_Render = data_loader_Render.load_data()
    iterator_Render = iter(dataset_Render)

    # model
    if model is None:
        model = create_model(opt)
    else:
        model.reset_train_mode(opt)
    V.show_options(opt, 'options')
    V.show_model_setting(model)
    V.save()

    # train
    checkpoints_dir = opt.checkpoints_dir
    IID_checkpoints_dir = os.path.join(checkpoints_dir, 'intrinsics/')

    train_epoch = opt.train_epoch
    iter_each_epoch = opt.iter_each_epoch
    num_iterations = dataset_size_CGIntrinsics / opt.batch_size_intrinsics
    criteria_label = opt.criteria

    best_eval = 0
    eval_saw = 1000000
    eval_iiw = 1000000
    best_epoch = 0
    best_model_file_path = None

    os_t = 0
    print_freq = 60
    for epoch in range(0, train_epoch):
        # learning rate
        V.show_learning_rate(model)

        loss = 0
        end = time.time()
        for i, data in enumerate(dataset_CGIntrinsics):
            if (iter_each_epoch is not None) and (i >= iter_each_epoch):
                break

            data_set_name = 'CGIntrinsics'
            inputs = {'input_srgb': data['img_1']}
            targets = data['target_1']
            loss += model.optimize(inputs, targets, criteria_label, data_set_name)

            # Optimize for small number of super high quality rendered images
            os_t += 1
            if os_t % 10 == 0:
                data_R = next(iterator_Render, None)
                if data_R is None:
                    iterator_Render = iter(dataset_Render)
                    data_R = next(iterator_Render, None)
                data_set_name = 'Render'
                inputs = {'input_srgb': data_R['img_1']}
                targets = data_R['target_1']
                loss += model.optimize(inputs, targets, criteria_label, data_set_name)

            if (i + 1) % print_freq == 0:
                loss /= float(print_freq)
                train_time = (time.time() - end) / print_freq
                print('%s: epoch %d, iteration %d/%d, loss %f' % (print_label, epoch, i, num_iterations, loss))
                print('    time: %.3lfs, batch %d(CGI) %d(IIW) %d(SAW)' %
                      (train_time, opt.batch_size_intrinsics, opt.batch_size_iiw, opt.batch_size_saw))
                V.vis.plot('lose', loss)
                loss = 0
                end = time.time()

        # Validation
        WHDR = 1.0
        if CriteriaTypes.train_reflectance(criteria_label):
            WHDR, _, _ = validate_iiw(model, opt, False,
                                      False, IID_checkpoints_dir+'validate/iiw/', label='val_iiw_%03d' % (epoch,),
                                      visualize_interval=60, use_subset=True)
        AP = 0.0
        if CriteriaTypes.train_shading(criteria_label):
            AP = validate_saw(model, full_root, False, 1, False, IID_checkpoints_dir+'validate/saw/',
                              label='val_saw_%03d' % (epoch,), samples=20, use_subset=True)
        if CriteriaTypes.train_shading(criteria_label) and not CriteriaTypes.train_reflectance(criteria_label):
            rel_eval = AP / tg_AP
        elif CriteriaTypes.train_reflectance(criteria_label) and not CriteriaTypes.train_shading(criteria_label):
            rel_eval = tg_WHDR / WHDR
        else:
            rel_eval = tg_WHDR / WHDR * 0.45 + AP / tg_AP * 0.55
        print('\nValidation for epoch %d:\n    WHDR %.5lf, AP %.5lf, rel_eval %.4lf' % (epoch, WHDR, AP, rel_eval))
        V.vis.plot_many({'Validation on iiw': WHDR,
                         'Validation on saw_subset': AP,
                         'Relative validation': rel_eval})
        model.set_evaluation_results(rel_eval)
        if rel_eval > best_eval:
            best_eval = rel_eval
            eval_iiw = WHDR
            eval_saw = AP
            best_epoch = epoch

        # Save checkpoint file
        saved_model_file_path = model.save(checkpoints_dir + 'train/', 'IID' + '_%03d' % epoch, epoch=epoch,
                                           iiw_val=WHDR, saw_val=AP, best_IID=(best_epoch == epoch))
        if best_epoch == epoch:
            best_model_file_path = saved_model_file_path
        print('%s: best epoch %d, eval iiw %.5lf, eval saw %.5lf, best eval %.4lf\n' %
              (print_label, best_epoch, eval_iiw, eval_saw, best_eval))
        V.save()

    return model, best_model_file_path


def train_intrinsic_image_decomposition(stage, params):
    # torch setting
    pytorch_settings.set_(with_random=True, determine=False)

    # Configuration
    opt = TrainIIDOptions()
    opt.parse(params)
    root_checkpoint_dir = opt.checkpoints_dir
    root_env = opt.env

    # Tree stages of training
    model = None

    # STAGE 1: train shading branch
    if stage <= 1:
        # warm_up:
        opt.optim_IID = 'full'  # optimize the IID-Net
        opt.checkpoints_dir = os.path.join(root_checkpoint_dir, 'stage_1_warm_up/')
        opt.env = root_env + '_stage_1_warm_up'  # visdom env
        opt.train_epoch = 30
        opt.lr = 1e-4
        opt.criteria = CriteriaTypes.warm_up
        model, best_model_file_path = _train_CGIntrinsics(model, opt, 'Stage_1_warm_up')
        opt.pretrained_file = best_model_file_path  # best model from previous training stage
        opt.load_pretrained_NEM = True
        opt.load_pretrained_IID_Net = True

        # stage_1:
        opt.optim_IID = 'wo_R'  # optimize the encoder and the shading decoder of the IID-Net
        opt.checkpoints_dir = os.path.join(root_checkpoint_dir, 'stage_1/')
        opt.env = root_env + '_stage_1'  # visdom env
        opt.train_epoch = 20
        opt.lr = 5e-5
        opt.criteria = CriteriaTypes.shading
        model, best_model_file_path = _train_CGIntrinsics(model, opt, 'Stage_1')
        opt.pretrained_file = best_model_file_path  # best model from previous training stage
        opt.load_pretrained_NEM = True
        opt.load_pretrained_IID_Net = True

    # STAGE 2: train R decoder
    opt.optim_IID = 'R'  # optimize the reflectance decoder of the IID-Net
    opt.checkpoints_dir = os.path.join(root_checkpoint_dir, 'stage_2/')
    opt.env = root_env + '_stage_2'  # visdom env
    opt.train_epoch = 5
    opt.lr = 1e-4
    opt.criteria = CriteriaTypes.reflectance
    if stage <= 2:
        model, best_model_file_path = _train_CGIntrinsics(model, opt, 'Stage_2')
        opt.pretrained_file = best_model_file_path  # best model from previous training stage
        opt.load_pretrained_NEM = True
        opt.load_pretrained_IID_Net = True

    # STAGE 3: finetune the IID-Net
    opt.optim_IID = 'full'  # optimize the IID-Net
    opt.checkpoints_dir = os.path.join(root_checkpoint_dir, 'stage_3/')
    opt.env = root_env + '_stage_3'  # visdom env
    opt.train_epoch = 30
    opt.iter_each_epoch = 500
    opt.lr = 2e-5
    opt.criteria = CriteriaTypes.IID
    if stage <= 3:
        model, best_model_file_path = _train_CGIntrinsics(model, opt, 'Stage_3')


def train_intrinsic_image_decomposition_simplified(stage, params):
    # torch setting
    pytorch_settings.set_(with_random=True, determine=False)

    # Configuration
    opt = TrainIIDOptions()
    opt.parse(params)
    root_checkpoint_dir = opt.checkpoints_dir
    root_env = opt.env

    # Stages of training
    model = None

    # STAGE 1:
    if stage <= 1:
        opt.parse({
            'optim_IID': 'full',
            'checkpoints_dir': os.path.join(root_checkpoint_dir, 'stage_A/'),
            'env': root_env + '_stage_A',  # visdom env
            'train_epoch': 25,
            'lr': 1e-4,
            'criteria': CriteriaTypes.warm_up
        })
        model, best_model_file_path = _train_CGIntrinsics(model, opt, 'Stage_A')
        opt.parse({
            'pretrained_file': best_model_file_path,  # best model from previous training stage
            'load_pretrained_NEM': True,
            'load_pretrained_IID_Net': True
        })

    # STAGE 2:
    if stage <= 2:
        opt.parse({
            'optim_IID': 'full',
            'checkpoints_dir': os.path.join(root_checkpoint_dir, 'stage_B/'),
            'env': root_env + '_stage_B',  # visdom env
            'train_epoch': 25,
            'lr': 5e-5,
            'criteria': CriteriaTypes.IID
        })
        model, best_model_file_path = _train_CGIntrinsics(model, opt, 'Stage_B')


if __name__ == '__main__':
    params = {
        'gpu_num': 4,  # i.e., 'gpu_devices' = [0, 1, 2, 3]
        # 'gpu_devices': [1, 2],
        'batch_size_intrinsics': 16,  # batch_size for CGIntrinsics data during training
        'batch_size_iiw': 16,  # batch_size for IIW data during validation
        'num_workers_intrinsics': 2,  # num_workers for data_loader
        'offline': True,  # set True if running offline visdom
        'pretrained_file': './pretrained_model/final.pth.tar',
        # 'load_pretrained_NEM': True,
        # 'load_pretrained_IID_Net': True,
        'checkpoints_dir': './checkpoints/train_IID-Net/',
        'env': 'train-IID-Net'
    }
    train_intrinsic_image_decomposition(1, params)
    # train_intrinsic_image_decomposition_simplified(1, params)
