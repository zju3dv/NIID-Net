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
import torch.sparse
# from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import utils.visualize as V


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()

    def forward(self, pred, target, valid_mask, weight_mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        diff = torch.mul(valid_mask, diff)
        diff = torch.pow(diff, 2)
        if weight_mask is not None:
            diff *= weight_mask
        num_valid = torch.sum(valid_mask)
        loss = torch.sum(diff)/(num_valid + 1e-6)
        return loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, valid_mask, weight_mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        diff = torch.mul(valid_mask, diff)
        diff = torch.abs(diff)
        if weight_mask is not None:
            diff *= weight_mask
        num_valid = torch.sum(valid_mask)
        loss = torch.sum(diff)/(num_valid + 1e-6)
        return loss


class LaplaceFilter_5D(nn.Module):
    def __init__(self):
        super(LaplaceFilter_5D, self).__init__()
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        edge = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ])
        edge_k = edge
        edge_k = torch.from_numpy(edge_k).to(torch.float32).view(1, 1, 5, 5)
        self.edge_conv.weight = nn.Parameter(edge_k, requires_grad=False)

        if True:
            self.mask_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
            mask_k = np.array([
                [0, 0, 0.077, 0, 0],
                [0, 0.077, 0.077, 0.077, 0],
                [0.077, 0.077, 0.077, 0.077, 0.077],
                [0, 0.077, 0.077, 0.077, 0],
                [0, 0, 0.077, 0, 0]
            ])
            mask_k = torch.from_numpy(mask_k).to(torch.float32).view(1, 1, 5, 5)
            self.mask_conv.weight = nn.Parameter(mask_k, requires_grad=False)

        for param in self.parameters():
            param.requires_grad = False

    def apply_laplace_filter(self, x, mask=None):
        out = self.edge_conv(x)
        if mask is not None:
            out_mask = self.mask_conv(mask)
            out_mask[out_mask < 0.95] = 0
            out_mask[out_mask >= 0.95] = 1
            out = torch.mul(out, out_mask)
        else:
            out_mask = None
        return out, out_mask

    def forward(self, x, mask=None):
        out, out_mask = self.apply_laplace_filter(x[:, 0:1, :, :], mask[:, 0:1, :, :] if mask is not None else None)
        for idx in range(1, x.size(1)):
            d_out, d_out_mask = self.apply_laplace_filter(x[:, idx:idx+1, :, :],
                                                          mask[:, idx:idx+1, :, :] if mask is not None else None)
            out = torch.cat((out, d_out), 1)
            if d_out_mask is not None:
                out_mask = torch.cat((out_mask, d_out_mask), 1)

        return out, out_mask


class L1ImageGradientLoss(nn.Module):
    def __init__(self, step=2):
        super(L1ImageGradientLoss, self).__init__()
        self.step = step

    def forward(self, pred, target, mask):
        step = self.step

        N = torch.sum(mask)
        diff = pred - target
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:, :, 0:-step, :] - diff[:, :, step:, :])
        v_mask = torch.mul(mask[:, :, 0:-step, :], mask[:, :, step:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, :, :, 0:-step] - diff[:, :, :, step:])
        h_mask = torch.mul(mask[:, :, :, 0:-step], mask[:, :, :, step:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient)) / 2.0
        gradient_loss = gradient_loss / (N + 1e-6)

        return gradient_loss


class SecondOrderGradLoss(nn.Module):
    def __init__(self):
        super(SecondOrderGradLoss, self).__init__()
        self.laplace = LaplaceFilter_5D()

    def forward(self, pred, target, mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions in SecondOrderGradLoss"
        lap_pred, mask_lap = self.laplace(pred, mask)
        lap_target, _ = self.laplace(target, mask)
        diff = (lap_pred - lap_target) * mask_lap
        tot_loss = torch.sum(torch.abs(diff)) / torch.sum(mask_lap + 1e-6)
        return tot_loss


class MultiScaleGradientLoss(nn.Module):
    def __init__(self, order=1, scale_step=2):
        super(MultiScaleGradientLoss, self).__init__()
        if order == 1:
            self.gradient_loss = L1ImageGradientLoss(step=1)
        elif order == 2:
            self.gradient_loss = SecondOrderGradLoss()
        self.step = scale_step

    def forward(self, pred, target, mask):
        step = self.step

        prediction_1 = pred[:,:,::step,::step]
        prediction_2 = prediction_1[:,:,::step,::step]
        prediction_3 = prediction_2[:,:,::step,::step]

        mask_1 = mask[:,:,::step,::step]
        mask_2 = mask_1[:,:,::step,::step]
        mask_3 = mask_2[:,:,::step,::step]

        gt_1 = target[:,:,::step,::step]
        gt_2 = gt_1[:,:,::step,::step]
        gt_3 = gt_2[:,:,::step,::step]

        final_loss = self.gradient_loss(pred, target, mask)
        final_loss += self.gradient_loss(prediction_1, gt_1, mask_1)
        final_loss += self.gradient_loss(prediction_2, gt_2, mask_2)
        final_loss += self.gradient_loss(prediction_3, gt_3, mask_3)
        return final_loss


class ReflectConsistentLoss(nn.Module):
    def __init__(self, sample_num_per_area=1, split_areas=(3, 3)):
        super(ReflectConsistentLoss, self).__init__()
        self.sample_num = sample_num_per_area
        self.split_x, self.split_y = split_areas

        self.cos_similar = nn.CosineSimilarity(dim=1, eps=0)

    def random_relative_loss(self, random_pixel, pred_R, gt_R, target_rgb, mask):
        x, y = random_pixel

        samples_gt_R = gt_R[:, :, x:x+1, y:y+1]
        samples_rgb = target_rgb[:, :, x:x+1, y:y+1]
        samples_pred_R = pred_R[:, :, x:x+1, y:y+1]
        samples_mask = mask[:, :, x:x+1, y:y+1]

        rel_gt_R = gt_R - samples_gt_R
        # rel_rgb = target_rgb - samples_rgb
        rel_pred_R = pred_R - samples_pred_R

        # Compute similarity
        mask_rel = mask * samples_mask
        mean_rel_gt_R = torch.mean(torch.abs(rel_gt_R), dim=1, keepdim=True).repeat(1, 3, 1, 1) * mask_rel
        mask_rel[mean_rel_gt_R >= 0.2] = 0

        diff = (rel_pred_R - rel_gt_R) * mask_rel
        loss = torch.sum(torch.abs(diff)) / (torch.sum(mask_rel)+1e-6)
        return loss

    def forward(self, pred_R, gt_R, target_rgb, mask):
        device = pred_R.device
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        x_spaces = np.linspace(0, gt_R.size(2), self.split_x+1, endpoint=True, dtype=np.int)
        y_spaces = np.linspace(0, gt_R.size(3), self.split_y+1, endpoint=True, dtype=np.int)
        for idx in range(self.sample_num):
            for idx_x in range(x_spaces.size - 1):
                for idx_y in range(y_spaces.size - 1):
                    x = np.random.randint(x_spaces[idx_x], x_spaces[idx_x+1], 1)[0]
                    y = np.random.randint(y_spaces[idx_y], y_spaces[idx_y+1], 1)[0]
                    total_loss += self.random_relative_loss((x, y), pred_R, gt_R, target_rgb, mask)
        return total_loss / (self.sample_num * self.split_x * self.split_y)


class LightingSmoothLoss(nn.Module):
    def __init__(self):
        super(LightingSmoothLoss, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.laplace_filter_5D = LaplaceFilter_5D()
        self.thrld_smooth_N = 0.05

    def forward(self, pred_L, N, mask, mode):
        L_length = torch.linalg.norm(pred_L, ord=2, dim=1, keepdim=True)
        N = N / torch.linalg.norm(N, ord=2, dim=1, keepdim=True).clamp(min=1e-6)

        # smooth light direction loss
        cos_h = self.cos(pred_L[:, :, :-1, :], pred_L[:, :, 1:, :])
        cos_v = self.cos(pred_L[:, :, :, :-1], pred_L[:, :, :, 1:])
        loss_direct = ((-cos_h).mean() + (-cos_v).mean()) / 2.0

        # smooth light length loss
        if mode == 0:
            mask_1ch = torch.ones_like(L_length)
        else:
            Lap_N, _ = self.laplace_filter_5D(N)
            mask_smooth_N = (torch.abs(Lap_N).mean(dim=1, keepdim=True) < self.thrld_smooth_N).to(torch.float32)
            mask_1ch = mask.min(dim=1, keepdim=True)[0] * mask_smooth_N

        step = 1
        v_mask = (mask_1ch[:, :, 0:-step, :] * mask_1ch[:, :, step:, :]).clamp(min=0.1)
        v_gradient = (L_length[:, :, 0:-step, :] - L_length[:, :, step:, :]).abs() * v_mask
        h_mask = (mask_1ch[:, :, :, 0:-step] * mask_1ch[:, :, :, step:]).clamp(min=0.1)
        h_gradient = (L_length[:, :, :, 0:-step] - L_length[:, :, :, step:]).abs() * h_mask
        loss_length = v_gradient.sum()/v_mask.sum() + h_gradient.sum()/h_mask.sum()
        loss_length /= 2.0

        # V.vis.img_many({
        #     'L int': L_length[0].clamp(0, 1).data.cpu(),
        #     'mask smooth': mask_1ch[0].data.cpu(),
        #     'pred N': ((-N+1)/2.0)[0].data.cpu(),
        #     'v_mask': v_mask[0].data.cpu(),
        #     'h_mask': h_mask[0].data.cpu()
        # })

        loss = loss_length + loss_direct * 0.1
        return loss


class CGIntrinsics_Criterion(nn.Module):
    def __init__(self):
        super(CGIntrinsics_Criterion, self).__init__()
        self.value_R_criterion = MaskedL1Loss()
        self.value_S_criterion = MaskedL1Loss()
        self.grad_R_criterion = MultiScaleGradientLoss(order=1, scale_step=4)
        self.grad_S_criterion = SecondOrderGradLoss()
        self.global_R_criterion = ReflectConsistentLoss(sample_num_per_area=1, split_areas=(1, 1))
        self.smooth_L_criterion = LightingSmoothLoss()
        self.w_R = 1.0
        self.w_grad_R = 9.0
        self.w_global_R = 9.0
        self.w_S = 1.0
        self.w_grad_S = 9.0
        self.w_smooth_L = 3.0
        self.cnt = 0

    def forward(self, input_images, N, R, L, S, targets, smooth_L_with_N, compute_loss_R, compute_loss_S):
        # GPU
        device = N.device
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)


        #### GroundTruth ####

        # gt_R = Variable(targets['gt_R'].float().cuda(device_id), requires_grad=False)
        # gt_S = Variable(targets['gt_S'].float().cuda(device_id), requires_grad=False)
        # rgb_img = Variable(targets['rgb_img'].float().cuda(device_id), requires_grad=False)
        # mask = Variable(torch.min(targets['mask'].float().cuda(device_id), dim=1, keepdim=True)[0], requires_grad=False)
        gt_R = targets['gt_R']
        gt_S = targets['gt_S']
        rgb_img = targets['rgb_img']
        mask = torch.min(targets['mask'], dim=1, keepdim=True)[0]

        # Same size
        size = [N.size(2), N.size(3)]
        gt_R = F.upsample(gt_R, size, mode='bilinear', align_corners=True)
        gt_S = F.upsample(gt_S, size, mode='bilinear', align_corners=True)
        gt_S_intensity = torch.mean(gt_S, dim=1, keepdim=True)  # shading intensity
        rgb_img = F.upsample(rgb_img, size, mode='bilinear', align_corners=True)
        mask = F.upsample(mask, size, mode='bilinear', align_corners=True)
        mask = (mask >= 0.999).to(torch.float32)

        #### Loss function ####

        if compute_loss_R:
            mask_R = mask.repeat(1, gt_R.size(1), 1, 1)
            mask_img = mask.repeat(1, input_images.size(1), 1, 1)
            R_loss = self.w_R * self.value_R_criterion(R, gt_R, mask_R)
            grad_R_loss = self.w_grad_R * self.grad_R_criterion(R, gt_R, mask_R)
            global_R_loss = self.w_global_R * self.global_R_criterion(R, gt_R, rgb_img, mask_img)
            total_loss += R_loss + grad_R_loss + global_R_loss

        if compute_loss_S:
            S_intensity = S
            L_intensity = L
            mask_S_intensity = mask.repeat(1, gt_S_intensity.size(1), 1, 1)
            mask_L_intensity = mask.repeat(1, L_intensity.size(1), 1, 1)
            # L Loss
            smooth_mode = 1 if smooth_L_with_N else 0
            smooth_L_loss = self.w_smooth_L * self.smooth_L_criterion(L_intensity, N, mask_L_intensity, smooth_mode)
            # S Loss
            S_loss = self.w_S * self.value_S_criterion(S_intensity, gt_S_intensity, mask_S_intensity)
            grad_S_loss = self.w_grad_S * self.grad_S_criterion(S_intensity, gt_S_intensity, mask_S_intensity)
            total_loss += S_loss + grad_S_loss + smooth_L_loss

        # visualize
        # if self.cnt % 30 == 0:
        #     L_len = torch.sum(L_intensity ** 2, dim=1, keepdim=True) ** 0.5
        #     L_direct = L_intensity / L_len.clamp(1e-6)
        #     V.vis.img_many({
        #         'rgb_img': rgb_img.cpu().data[0, :, :, :],
        #         'gt_R': gt_R.cpu().data[0, :, :, :],
        #         'gt_S_intensity': torch.clamp(gt_S_intensity.cpu().data[0, 0, :, :], 0, 1),
        #         # 'gt_mask': mask_img.cpu().data[0, :, :, :],
        #         'pred_N': ((-N[0, :, :, :] + 1.0) / 2.0).data.cpu().clamp(0, 1),
        #         # 'pred_R': torch.clamp(R.data.cpu()[0, :, :, :], 0, 1),
        #         'pred_S_intensity': torch.clamp(S_intensity[0, :, :, :].data.cpu(), 0, 1),
        #         'pred_L_direction': ((-L_direct[0, :, :, :] + 1.0) / 2.0).data.cpu(),
        #         'pred_L_length': (L_len / (torch.max(L_len) + 1e-6)).data.cpu()[0, :, :, :],
        #     })
        #     self.cnt = 0
        # self.cnt += 1

        return total_loss


class JointCriterion(nn.Module):
    def __init__(self):
        super(JointCriterion, self).__init__()
        self.w_intrinsics = 1.0
        self.w_saw = 1.0
        self.w_iiw = 1.0
        self.CGIntrinsics_criterion = CGIntrinsics_Criterion()
        self.total_loss = None

    def forward(self, input_images, N, R, L, S, targets, data_set_name):
        if data_set_name == "IIW":
            pass
        elif data_set_name == "CGIntrinsics" or data_set_name == "Render":
            return self.w_intrinsics * self.CGIntrinsics_criterion(input_images, N, R, L, S, targets)
        elif data_set_name == "SAW":
            pass
        else:
            raise Exception('Not support dataset [%s]!' % data_set_name)
        return None

