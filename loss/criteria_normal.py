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

import utils.visualize as V
from loss.criteria_intrinsic import MultiScaleGradientLoss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = (target - pred).abs() * valid_mask
        loss = torch.sum(diff) / torch.sum(valid_mask)
        return loss


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()

    def forward(self, pred, target, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = (target - pred)**2 * valid_mask
        loss = torch.sum(diff) / torch.sum(valid_mask)
        return loss


class CNT(object):
    num = 0


class NormalCriterion(nn.Module):
    def __init__(self):
        super(NormalCriterion, self).__init__()
        self.value_loss = MaskedL1Loss()
        self.gradient_loss = MultiScaleGradientLoss(order=1, scale_step=4)
        self.w_gradient = 1.0
        self.cnt = CNT()

    def normalize_surface_normal(self, normal, valid_mask):
        length = torch.sum(normal ** 2, dim=1, keepdim=True) ** 0.5
        # valid_mask[length < 1e-6] = 0.0
        length[length < 1e-6] = 1e-6
        normal = (normal / length) * valid_mask
        return normal, valid_mask

    def visualize(self, pred_N, gt_N, mask, input_srgb):
        if self.cnt.num % 50 == 0:
            vis_N = pred_N / torch.linalg.norm(pred_N, ord=2, dim=1, keepdim=True).clamp(min=1e-6)
            V.vis.img_many({
                'pred_normal': (-vis_N.data.cpu()[0]+1.0)/2.0,
                'gt_normal': (-gt_N.data.cpu()[0]+1.0)/2.0,
                'valid_normal': mask.data.cpu()[0],
                'input_srgb': input_srgb.data.cpu()[0],
            })
            self.cnt.num = 0
        self.cnt.num += 1

    def forward(self, pred_N, gt_N, mask, input_srgb):
        # same size
        t_size = [pred_N.size(2), pred_N.size(3)]
        gt_N = F.upsample(gt_N, size=t_size, mode='bilinear', align_corners=True)
        mask = F.upsample(mask, size=t_size, mode='bilinear', align_corners=True).repeat(1, pred_N.size(1), 1, 1)
        mask = (mask >= 0.99).to(torch.float32)
        total_loss = self.value_loss(pred_N, gt_N, mask)\
                     + self.w_gradient * self.gradient_loss(pred_N, gt_N, mask)
        # visualize
        # self.visualize(pred_N, gt_N, mask, input_srgb)
        return total_loss