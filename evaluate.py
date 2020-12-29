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

import torch
import numpy as np
import h5py
from math import ceil

from config import TestOptions
from models.manager import create_model
from test import metrics_iiw, metrics_saw
from utils import image_util
from utils import pytorch_settings
import utils.visualize as V


def validate_iiw(model, opt, use_test_split,
                 display_process=True, visualize_dir=None, label='val_iiw', visualize_interval=1, use_subset=False):
    """Evaluate reflectance estimation of the model

    :param model:
    :param opt:
        configuration
    :param use_test_split:
        True: test split
        False: train split
    :param display_process:
        print the evaluation process or not
    :param visualize_dir:
        directory for visualized results
    :param label:
        prefix of the name of the output file
    :param visualize_interval:
        visualize one sample every [visualize_interval] input
    :param use_subset:
        evaluate on a subset or not
    """
    
    print("============================= Validation ON IIW============================")
    print('batch size in validate_iiw: iiw %d' % opt.batch_size_iiw)

    full_root = opt.dataset_root
    list_name = 'test_list/' if use_test_split else 'train_list/'

    total_loss =0.0
    total_loss_eq =0.0
    total_loss_ineq =0.0
    total_count = 0.0

    from data.intrinsics import data_loader as intrinsics_data_loader

    cnt = 0
    num_batch = ceil(visualize_interval / opt.batch_size_iiw)
    # for 3 different orientation
    for j in range(0, 3):
        test_list_dir = full_root + '/CGIntrinsics/IIW/' + list_name
        data_loader_iiw_TEST = intrinsics_data_loader.CreateDataLoaderIIWTest(full_root, test_list_dir, j,
                                                                              _batch_size=opt.batch_size_iiw,
                                                                              _num_workers=opt.num_workers_intrinsics,
                                                                              use_subset=use_subset)
        dataset_iiw_test = data_loader_iiw_TEST.load_data()

        for i, data_iiw in enumerate(dataset_iiw_test):
            inputs = {'input_srgb': data_iiw['img_1']}
            targets = data_iiw['target_1']

            pred_N, pred_R, pred_L, pred_S, rendered_img = model.predict(inputs, normal=True, IID=True)
            total_whdr, total_whdr_eq, total_whdr_ineq, count = \
                metrics_iiw.evaluate_WHDR(image_util.rgb_to_srgb(pred_R), targets)
            total_loss += total_whdr
            total_loss_eq += total_whdr_eq
            total_loss_ineq += total_whdr_ineq

            total_count += count
            if display_process:
                print("Testing WHDR ", j, i, total_loss/total_count)

            cnt += 1
            if visualize_dir is not None:
                if cnt % num_batch == 0:
                    idx = 0
                    pred_imgs = {
                        'pred_N': pred_N[idx].cpu(),
                        'pred_R': pred_R[idx].cpu(),
                        'pred_L': pred_L[idx].cpu(),
                        'pred_S': pred_S[idx].cpu(),
                        'rendered_img': rendered_img[idx].cpu(),
                        'input_srgb': inputs['input_srgb'][idx].cpu(),
                    }
                    # pred_imgs = {k: eval_results[k][idx].cpu()
                    #              for k in eval_results.keys() if torch.is_tensor(eval_results[k])}
                    image_util.save_intrinsic_images(visualize_dir, pred_imgs, label='%s_%s-%s-%s' % (label, j, i, idx))

    return total_loss/(total_count), total_loss_eq/total_count, total_loss_ineq/total_count


def test_IIW(**kwargs):
    """ Evaluate reflectance estimation on the IIW test set

    :param kwargs:
        configuration that need to be updated
    """

    # parse parameters
    opt = TestOptions()
    opt.parse(kwargs)
    # print(kwargs)
    output_dir = opt.checkpoints_dir + 'test_iiw/'
    visualize_interval = 40

    # torch setting
    pytorch_settings.set_(with_random=False, determine=True)

    # visualize
    V.create_a_visualizer(opt)

    # Model Manager
    model = create_model(opt)
    model.switch_to_eval()

    WHDR, WHDR_EQ, WHDR_INEQ = validate_iiw(model, opt, True,
                                            True, output_dir, label='test_iiw', visualize_interval=visualize_interval,
                                            use_subset=False)
    print('Test IIW: WHDR %f' % WHDR)


def save_plot_arr(path, plot_arr):
    hdf5_file_write = h5py.File(path, 'w')
    hdf5_file_write.create_dataset('plot_arr', data=plot_arr)
    hdf5_file_write.close()


def validate_saw(model, full_root, use_test_split, mode,
                 display_process=True, visualize_dir=None, label='val_saw', samples=0, use_subset=False):
    """ Evaluate shading estimation of the model

    :param model:
    :param full_root:
        root directory for datasets
    :param use_test_split:
        True: test split
        False: train split
    :param mode:
        0 : unweighted precision (P(u))
        1 : challenge precision (P(c))
    :param display_process:
        print the evaluation process or not
    :param visualize_dir:
        directory for the PR_array and visualized results
    :param label:
        prefix of the name of the output file
    :param samples:
        number of samples that will be visualized and saved
    :param use_subset:
        evaluate on a subset or not
    :return:
        AP result
    """

    print("============================ Validation ON SAW============================")

    # parameters for SAW
    pixel_labels_dir = full_root + \
                       'CGIntrinsics/SAW/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = full_root + 'CGIntrinsics/SAW/saw_splits/'
    img_dir = full_root + "CGIntrinsics/SAW/saw_images_512/"
    dataset_split = 'E' if use_test_split else 'R'
    class_weights = [1, 1, 2]
    bl_filter_size = 10

    AP, plot_arr, sample_arr = metrics_saw.compute_pr(model.predict_IID_np_for_saw_eval,
                                                      pixel_labels_dir, splits_dir,
                                                      dataset_split, class_weights, bl_filter_size,
                                                      img_dir,
                                                      mode=mode, display_process=display_process, samples=samples,
                                                      use_subset=use_subset)

    # Save visualized results of samples and PR_array
    if visualize_dir is not None:
        for idx, result in enumerate(sample_arr):
            pred_imgs = {k: torch.from_numpy(np.transpose(result[k], (2, 0, 1))).contiguous().float()
                         for k in result.keys()}
            # pred_imgs = {
            #     'pred_N': torch.from_numpy(np.transpose(result['pred_N'], (2, 0, 1))).contiguous().float(),
            #     'pred_R': torch.from_numpy(np.transpose(result['pred_R'], (2, 0, 1))).contiguous().float(),
            #     'pred_L': torch.from_numpy(np.transpose(result['pred_L'], (2, 0, 1))).contiguous().float(),
            #     'pred_S': torch.from_numpy(np.transpose(result['pred_S'], (2, 0, 1))).contiguous().float(),
            #     'input_srgb': torch.from_numpy(np.transpose(result['input_srgb'], (2, 0, 1))).contiguous().float(),
            #     'rendered_img': torch.from_numpy(np.transpose(result['rendered_img'], (2, 0, 1))).contiguous().float(),
            # }
            image_util.save_intrinsic_images(visualize_dir, pred_imgs, '%s_%d-%d' % (label, idx, samples), False)
        # save PR_array
        if mode == 0:
            file_name = 'plot_arr_u.h5'
        else:
            file_name = 'plot_arr_c.h5'
        save_plot_arr(os.path.join(visualize_dir, '%s_%s' % (label, file_name)), plot_arr)
        print('save plot_arr: %s (%s)' % (os.path.join(visualize_dir, file_name), str(plot_arr.shape)))
    return AP


def test_SAW(mode=1, **kwargs):
    """ Evaluate shading estimation on the SAW test set

    :param mode:
        0 : unweighted precision (P(u))
        1 : challenge precision (P(c))
    :param kwargs:
        configuration that need to be updated
    """

    # parse parameters
    opt = TestOptions()
    opt.parse(kwargs)
    # print(kwargs)
    output_dir = os.path.join(opt.checkpoints_dir, 'test_saw/')
    output_label = 'test_saw'
    num_visualized_sample = 80

    # torch setting
    pytorch_settings.set_(with_random=False, determine=True)

    # visualize
    V.create_a_visualizer(opt)

    # Model Manager
    model = create_model(opt)
    model.switch_to_eval()

    AP = validate_saw(model, opt.dataset_root, True, mode, True, output_dir, output_label,
                      samples=num_visualized_sample, use_subset=False)
    print("Test SAW mode %d: AP %f" % (mode, AP))


if __name__ == '__main__':
    params = {
        'pretrained_file': './pretrained_model/final.pth.tar',
        'offline': True,
        'batch_size_iiw': 8,
        'num_workers_intrinsics': 1,
        'gpu_devices': [0,],
    }

    test_IIW(**params)
    test_SAW(mode=1, **params)
