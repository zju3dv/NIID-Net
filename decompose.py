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
from os import listdir
import argparse
import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config import TestOptions
from models.manager import create_model
from utils import image_util
from utils.image_util import convert_to_common_coordinate_system
from utils import pytorch_settings
import utils.visualize as V


def list_files(directory, extension):
    """ List files with specified suffixes in the directory (exclude subdirectories)
    """
    file_list = listdir(directory)
    included_list = []
    for f in file_list:
        for ext in extension:
            if f.endswith('.' + ext):
                included_list.append(f)
                break
    return included_list


def resize_image(img, img_size):
    """ Resize an image into the specified size

    :param img (torch.Tensor)
    :param img_size (int or tuple/list of 2 integers)
    :return: resized image (torch.Tensor)
    """
    DIVISOR = 32
    if img_size is None:
        return img
    assert torch.is_tensor(img), 'The input image should be a tensor'
    assert img.ndim == 4, 'The input image should be 4D tensor: B X C X H X W'
    if isinstance(img_size, int):
        # Resize the maximum side to the specified size
        ratio = img_size / max(img.shape[2], img.shape[3])
        t_h = img.shape[2] * ratio
        t_w = img.shape[3] * ratio
    elif isinstance(img_size, tuple) or isinstance(img_size, list):
        assert len(img_size) == 2, "img_size should be a tuple or list of 2 elements"
        t_h, t_w = img_size
    else:
        raise ValueError("img_size should be an integer or tuple/list of 2 integers")
    t_h = math.ceil(t_h / DIVISOR) * DIVISOR
    t_w = math.ceil(t_w / DIVISOR) * DIVISOR
    scaled_img = torch.nn.functional.interpolate(img, size=[t_h, t_w], mode='bilinear', align_corners=True)
    return scaled_img


def decompose_images(data_dir, output_dir, save_individually, resize_to_specified_size, save_srgb, **kwargs):
    """ Decompose all the images in a directory

    :param data_dir:
        the directory for input images
    :param output_dir:
        output directory
    :param save_individually:
        save intrinsic images individually or not
    :param resize_to_specified_size:
        resize input images to the specified size. Integer or tuple/list of 2 integers (H X W)
    :param save_srgb:
        save intrinsic images in the sRGB space or not
    :param kwargs:
        parameters for the NIID-Net and the visdom visualizer
    """

    # parse parameters
    opt = TestOptions()
    opt.parse(kwargs)
    # print(kwargs)

    # torch setting
    pytorch_settings.set_(with_random=False, determine=True)

    # visualize
    V.create_a_visualizer(opt)

    # NIID-Net Manager
    model = create_model(opt)
    model.switch_to_eval()

    # List all image files in the directory (exclude subdirectory)
    image_file_list = list_files(data_dir, ['jpg', 'jpeg', 'png', 'tif', 'JPG'])
    print('Total image in the directory %s: %d' % (data_dir, len(image_file_list)))

    # Decompose images
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    for file_name in image_file_list:
        # Read image
        img_path = os.path.join(data_dir, file_name)
        o_img = Image.open(img_path)
        o_img = o_img.convert("RGB")

        # Resize input image
        input_img = TF.to_tensor(o_img).to(torch.float32).unsqueeze(0)
        input_img = resize_image(input_img, resize_to_specified_size)

        # Predict
        # if set resize_input=True, the input image will be resized to a default size
        pred_N, pred_R, pred_L, pred_S, rendered_img = model.predict({'input_srgb': input_img}, normal=True, IID=True,
                                                                     resize_input=resize_to_specified_size is None)

        # Save results
        idx = 0
        pred_imgs = {
            'pred_N': pred_N[idx],
            'pred_R': pred_R[idx],
            'pred_L': pred_L[idx],
            'pred_S': pred_S[idx],
            'rendered_img': rendered_img[idx],
            'input_srgb': input_img[idx].to(pred_N.device),
        }
        for k in ["pred_N", "pred_L"]:
            pred_imgs[k] = convert_to_common_coordinate_system(pred_imgs[k])
        f = '%s' % (file_name[:file_name.rfind('.')])
        image_util.save_intrinsic_images(output_dir, pred_imgs, label=f, separate=save_individually,
                                         save_srgb=save_srgb)
        # torch.save(pred_imgs, os.path.join(output_dir, f+'.pth.tar'))
        raw_data = {
            k: v for k, v in pred_imgs.items() if k in ['pred_N', 'pred_R', 'pred_S']
        }
        for k, d in raw_data.items():
            np.save(os.path.join(raw_dir, f+'_'+k+'.npy'), d.detach().cpu().numpy().transpose(1, 2, 0))
        print('Decompose %s successfully!' % file_name)


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Decompose images using a pretrained model.")

    parser.add_argument('--data_dirs', nargs='+', default=['examples'],
                        help='List of input directories containing images to be decomposed.')
    parser.add_argument('--output_dir', default="out/", help='Directory where the output images will be saved.')
    parser.add_argument('--pretrained_file', default='./pretrained_model/final.pth.tar',
                        help='Path to the pretrained model file.')
    parser.add_argument('--online', action='store_true',
                        default=False,
                        help='Run Visdom in online mode.')
    parser.add_argument('--gpu_devices', type=int, nargs='+', default=[0],
                        help='List of GPU devices to use.')
    parser.add_argument('--not_save_individually', action='store_true', default=False,
                        help='Do not save intrinsic images individually.')
    parser.add_argument("--resize_to_specified_size", type=int, nargs='+', default=None,
                        help="Resize input images to the specified size. Integer or tuple/list of 2 integers (H X W).")
    parser.add_argument("--save_srgb", action='store_true', default=False,
                        help="Visualize the intrinsic images in the sRGB space.")
    args = parser.parse_args()
    if args.resize_to_specified_size is None:
        pass
    elif len(args.resize_to_specified_size) == 1:
        args.resize_to_specified_size = args.resize_to_specified_size[0]
    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    for data_dir in args.data_dirs:
        decompose_images(data_dir,  # input directory
                         os.path.join(args.output_dir, os.path.basename(data_dir)),  # output directory
                         not args.not_save_individually,  # whether to save intrinsic image as separate images
                         args.resize_to_specified_size,  # resize input images to the specified size
                         args.save_srgb,  # whether to save intrinsic images in the sRGB space
                         **{
                             'pretrained_file': args.pretrained_file,  # path to the pretrained model
                             'offline': not args.online,  # whether to run visdom in offline mode
                             'gpu_devices': args.gpu_devices,  # list of GPU devices to use
                            }
                         )
