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

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def rgb_to_srgb(rgb):
    ret = torch.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = torch.pow(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def srgb_to_rgb(srgb):
    ret = torch.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    sum = torch.sum(rgb, dim=0, keepdim=True) + 1e-6
    chromat = rgb / sum
    return chromat


def save_srgb_image(image, path, filename):
    # Transform to PILImage
    image_np = np.transpose(image.to(torch.float32).cpu().numpy(), (1, 2, 0)) * 255.0
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(image_np, mode='RGB')
    # Save Image
    if not os.path.exists(path):
        os.makedirs(path)
    image_pil.save(os.path.join(path,filename))


def adjust_image_for_display(image, rescale=True, trans2srgb=False, mask=None):
    MAX_SRGB = 1.077837  # SRGB 1.0 = RGB 1.077837

    vis = image.clone()
    if rescale:
        s = np.percentile(vis.cpu(), 99.9)
        # if mask is None:
        #     s = np.percentile(vis.numpy(), 99.9)
        # else:
        #     s = np.percentile(vis[mask > 0.5].numpy(), 99.9)
        if s > MAX_SRGB:
            vis = vis / s * MAX_SRGB

    vis = torch.clamp(vis, min=0)
    # vis[vis < 0] = 0
    if trans2srgb:
        vis[vis > MAX_SRGB] = MAX_SRGB
        vis = rgb_to_srgb(vis)
    if vis.size(0) == 1:
        vis = vis.repeat(3, 1, 1)
    return vis


def save_intrinsic_images(path, pred_images, label=None, separate=False):
    """ Visualize and save intrinsic images

    :param path: output directory
    :param images: images to be visualized
    :param label: prefix for output files
    :param separate: save visualized images separately or not
    """
    # Visualization for intrinsic images
    ## surface normal
    vis_N = (-pred_images['pred_N']+1) / 2.0

    ## reflectance
    pred_R = pred_images['pred_R']
    vis_R = adjust_image_for_display(pred_R, rescale=False, trans2srgb=True)
    # if pred_R.size(0) == 1:
    #     input_srgb = pred_images['input_srgb']
    #     rgb = srgb_to_rgb(input_srgb)
    #     chromat = rgb_to_chromaticity(rgb)
    #     pred_R = torch.mul(pred_R, chromat)

    ## integrated lighting
    pred_L = pred_images['pred_L']
    L_length = torch.sum(pred_L**2, dim=0, keepdim=True) ** 0.5
    L_direct = pred_L / (L_length + 1e-6)
    vis_L_length = adjust_image_for_display(L_length, rescale=True, trans2srgb=True)
    vis_L_direct = (-L_direct+1)/2.0
    vis_dot_NL = F.cosine_similarity(pred_images['pred_N'], pred_images['pred_L'], dim=0).unsqueeze(0)
    vis_dot_NL = adjust_image_for_display(vis_dot_NL, rescale=False, trans2srgb=True)

    ## shading
    vis_S = adjust_image_for_display(pred_images['pred_S'], rescale=True, trans2srgb=True)

    ## reconstructed image
    vis_rendered_img = adjust_image_for_display(pred_images['rendered_img'], rescale=False, trans2srgb=True)

    ## input image
    vis_srgb = pred_images['input_srgb']

    # Save intrinsic images
    if separate:
        save_srgb_image(vis_N, path, label+'_N.png')
        save_srgb_image(vis_R, path, label+'_R.png')
        save_srgb_image(vis_L_length, path, label+'_L_length.png')
        save_srgb_image(vis_L_direct, path, label+'_L_direct.png')
        save_srgb_image(vis_dot_NL, path, label+'_dotNL.png')
        save_srgb_image(vis_S, path, label+'_S.png')
        save_srgb_image(vis_srgb, path, label+'_rgb.png')
    else:
        # concatenate visualized images into one canva
        vis_merge_h1 = torch.cat((vis_srgb, vis_R, vis_S), 2)
        vis_merge_h2 = torch.cat((vis_L_direct, vis_L_length, vis_dot_NL), 2)
        vis_merge_h3 = torch.cat((vis_N, vis_rendered_img, torch.zeros_like(vis_N)), 2)
        vis_merge = torch.cat((vis_merge_h1, vis_merge_h2, vis_merge_h3), 1)
        save_srgb_image(vis_merge, path, label+'_result.png')


def save_normal_images(path, filename, input_srgb, normal_gt, normal_pred, normal_mask=None):
    # Input RGB
    srgb = input_srgb.cpu()
    srgb = 255 * np.transpose(np.squeeze(srgb.numpy()), (1, 2, 0))  # H, W, C

    # Normal target
    normal_gt = 255 * (-normal_gt+1) / 2.0
    if normal_mask is not None:
        normal_gt *= normal_mask
    normal_gt = np.transpose(normal_gt.cpu().numpy(), (1, 2, 0))

    # Normal prediction
    normal_pred = normal_pred / torch.linalg.norm(normal_pred, ord=2, dim=0, keepdim=True).clamp(min=1e-6)
    normal_pred = 255 * (-normal_pred+1) / 2.0
    normal_pred = np.transpose(normal_pred.cpu().numpy(), (1, 2, 0))

    # Merge
    img_merge = np.hstack([srgb, normal_gt, normal_pred])

    # Save image
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, filename)
    img_merge.save(filename)






