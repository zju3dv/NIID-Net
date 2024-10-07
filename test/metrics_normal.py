# reference: http://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html

import torch
# from torch.autograd import Variable
import torch.nn.functional as F
import math

from utils.visualize import Visualizer
# vis = Visualizer('vis_metrics_normal')


class SurfaceNormalMeter(object):
    eps = 2.2204e-16
    r1125 = math.radians(11.25)
    r2250 = math.radians(22.50)
    r3000 = math.radians(30.00)
    cos = torch.nn.CosineSimilarity(dim=1)

    def __init__(self):
        self.theta_collect = []

    def normalize(self, normal):
        length = torch.sum(normal ** 2, dim=1, keepdim=True) ** 0.5 + self.eps
        normal = normal / length
        return normal

    def add_evaluation(self, pred, gt, mask):
        assert pred.dim() == gt.dim(), 'Error: the dimension of pred and gt are not consistent!'

        # if isinstance(pred, Variable):
        #     pred = pred.data
        # if isinstance(gt, Variable):
        #     gt = gt.data
        # if isinstance(mask, Variable):
        #     mask = mask.data

        pred = pred.detach().to(torch.float64)  # numeric reason
        gt = gt.detach().to(torch.float64)
        mask = mask.detach().to(torch.float64)

        # pred = pred.double()
        # gt = gt.double()
        # mask = mask.double()
        # pred = self.normalize(pred)
        # gt = self.normalize(gt)

        theta = torch.acos(self.cos(pred, gt).unsqueeze(1)) * mask
        # theta = torch.acos(torch.clamp(torch.sum(pred*gt, dim=1, keepdim=True), min=-1.0, max=1.0)) * mask
        self.theta_collect.append(theta[mask > 0.5].cpu())

    def get_results(self):
        theta = torch.cat(self.theta_collect, dim=0)

        return {
            'mean': math.degrees(torch.mean(theta).item()),
            'median': math.degrees(torch.median(theta).item()),
            'rmse': math.degrees(torch.mean(theta**2).item()**0.5),
            'p1125': torch.mean((theta < self.r1125).to(torch.float32)).item(),
            'p2250': torch.mean((theta < self.r2250).to(torch.float32)).item(),
            'p3000': torch.mean((theta < self.r3000).to(torch.float32)).item(),
        }