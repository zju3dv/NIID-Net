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

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import iid_net
import models.Hu_nets.depth_net_from_Hu as normal_estimation_module


class NIIDNet(nn.Module):
    _imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}

    def __init__(self):
        super(NIIDNet, self).__init__()
        # Define Model
        self.IID_model = iid_net.IIDNet(3, (256, 512, 1024), 3, 3)
        self.NEM_coarse_model, self.NEM_refine_model = normal_estimation_module.build_model(
            is_resnet=False,
            is_densenet=False,
            is_senet=True,
        )
        self.IID_model.apply(iid_net.weights_init)

        # Input pre-processing parameters
        self.stats_mean = nn.Parameter(
            torch.Tensor(self._imagenet_stats['mean']).float().view(1, 3, 1, 1), requires_grad=False)
        self.stats_std = nn.Parameter(
            torch.Tensor(self._imagenet_stats['std']).float().view(1, 3, 1, 1), requires_grad=False)

    def resize_input_img(self, input_img):
        ratio = float(input_img.size(2)) / float(input_img.size(3))
        if ratio > 1.73:
            h, w = 320, 160 #512, 256
        elif ratio < 1.0 / 1.73:
            h, w = 160, 320 #256, 512
        elif ratio > 1.41:
            h, w = 384, 256
        elif ratio < 1. / 1.41:
            h, w = 256, 384
        elif ratio > 1.15:
            h, w = 320, 240 #512, 384
        elif ratio < 1. / 1.15:
            h, w = 240, 320 #384, 512
        else:
            h, w = 320, 320 #384, 384
        scaled_input_srgb = F.upsample(input_img, size=[h, w], mode='bilinear')
        return scaled_input_srgb

    def render(self, N, L, R=None):
        # shading intensity
        N = N / torch.norm(N, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        shading_intensity = F.relu(torch.sum(N * L, dim=1, keepdim=True))

        # render
        rendered_out = shading_intensity
        if R is not None:
            rendered_out = rendered_out * R

        return shading_intensity, rendered_out

    def forward(self, input_srgb, pred_normal, pred_reflect, pred_shading):
        if pred_reflect or pred_shading:
            pred_normal = True

        o_h = input_srgb.size(2)
        o_w = input_srgb.size(3)
        scaled_input_srgb = self.resize_input_img(input_srgb)

        out_N = None
        if pred_normal:
            normalized_scaled_input = (scaled_input_srgb - self.stats_mean) / self.stats_std
            (xb1, xb2, xb3, xb4), NEM_coarse_decoder = self.NEM_coarse_model(normalized_scaled_input)
            out_N = self.NEM_refine_model(NEM_coarse_decoder, xb1, xb2, xb3, xb4)
            out_N = out_N * 2.0 - 1.0  # range: [-1, 1]
            out_N = F.upsample(out_N, size=[o_h, o_w], mode='bilinear')

        out_R = out_L = out_S = rendered_img = None
        if pred_reflect or pred_shading:
            out_R, out_L = self.IID_model(scaled_input_srgb, (xb1, xb2, xb3),
                                          pred_reflect, pred_shading)
            if pred_reflect:
                out_R = torch.exp(out_R)
                out_R = F.upsample(out_R, size=[o_h, o_w], mode='bilinear')
            if pred_shading:
                out_L = F.upsample(out_L, size=[o_h, o_w], mode='bilinear')
                out_S, rendered_img = self.render(out_N, out_L, out_R)

        return out_N, out_R, out_L, out_S, rendered_img

