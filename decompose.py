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

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config import TestOptions
from models.manager import create_model
from utils import image_util
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


def resize_image(img):
    """ Resize an image into the specified scale according to its aspect ratio

    :param img (PIL.Image.Image)
    :return: resized image (PIL.Image.Image)
    """
    ratio = float(img.size[1]) / float(img.size[0])

    if ratio > 1.73:
        h, w = 320, 160  # 512, 256
    elif ratio < 1.0 / 1.73:
        h, w = 160, 320  # 256, 512
    elif ratio > 1.41:
        h, w = 384, 256
    elif ratio < 1. / 1.41:
        h, w = 256, 384
    elif ratio > 1.15:
        h, w = 320, 240  # 512, 384
    elif ratio < 1. / 1.15:
        h, w = 240, 320  # 384, 512
    else:
        h, w = 320, 320  # 384, 384

    h = int(h)
    w = int(w)
    img = TF.resize(img, size=[h, w], interpolation=Image.BILINEAR)
    return img


def decompose_images(data_dir, output_dir, save_individually, **kwargs):
    """ Decompose all the images in a directory

    :param data_dir:
        the directory for input images
    :param output_dir:
        output directory
    :param save_individually:
        save intrinsic images individually or not
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
    for file_name in image_file_list:
        # Read image
        img_path = os.path.join(data_dir, file_name)
        o_img = Image.open(img_path)
        o_img = o_img.convert("RGB")

        # Resize input image
        # input_img = resize_image(o_img)
        input_img = o_img

        # Predict
        input_img = TF.to_tensor(input_img).unsqueeze(0)
        pred_N, pred_R, pred_L, pred_S, rendered_img = model.predict({'input_srgb': input_img}, normal=True, IID=True)

        # Save results
        idx = 0
        pred_imgs = {
            'pred_N': pred_N[idx].cpu(),
            'pred_R': pred_R[idx].cpu(),
            'pred_L': pred_L[idx].cpu(),
            'pred_S': pred_S[idx].cpu(),
            'rendered_img': rendered_img[idx].cpu(),
            'input_srgb': input_img[idx],
        }
        f = '%s_decomposed' % (file_name[:file_name.rfind('.')])
        image_util.save_intrinsic_images(output_dir, pred_imgs, label=f, individual=save_individually)
        torch.save(pred_imgs, os.path.join(output_dir, f+'.pth.tar'))
        print('Decompose %s successfully!' % file_name)


if __name__ == '__main__':
    data_dirs = []
    data_dirs.append('examples/')

    for data_dir in data_dirs:
        decompose_images(data_dir,  # input directory
                         os.path.join(data_dir, 'decomposition_results'),  # output directory
                         False,  # whether to save intrinsic image as separate images
                         **{
                             'pretrained_file': './pretrained_model/final.pth.tar',
                             'offline': True,
                             'gpu_devices': [0],
                            }
                         )
