import torch
from torch import nn
import numpy as np


# SOURCE:
# https://github.com/xanderchf/MonoDepth-FPN-PyTorch/blob/master/main_fpn.py
def rmse_log(pred, target):
    loss = torch.sqrt(torch.mean(torch.abs(torch.log(target) - torch.log(pred)) ** 2))
    return loss


def l1(pred, target):
    if not pred.shape == target.shape:
        _, _, H, W = target.shape
        pred = F.upsample(pred, size=(H, W), mode="bilinear")
    loss = torch.mean(torch.abs(10.0 * target - 10.0 * pred))
    return loss


def l1_log(pred, target):
    if not pred.shape == target.shape:
        _, _, H, W = target.shape
        pred = F.upsample(pred, size=(H, W), mode="bilinear")
    loss = torch.mean(torch.abs(torch.log(target) - torch.log(pred)))
    return loss


def rmse(pred, target):
    loss = torch.sqrt(torch.mean(torch.abs(10.0 * target - 10.0 * pred) ** 2))
    return loss


def grad_loss(grad_pred, grad_target):
    return torch.sum(torch.mean(torch.abs(grad_target - grad_pred)))


def normal_loss(grad_pred, grad_target):
    prod = (
        (grad_pred[:, :, None, :] @ grad_target[:, :, :, None]).squeeze(-1).squeeze(-1)
    )
    pred_norm = torch.sqrt(torch.sum(grad_pred ** 2, dim=-1))
    target_norm = torch.sqrt(torch.sum(grad_target ** 2, dim=-1))
    return 1 - torch.mean(prod / (pred_norm * target_norm))


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


# class BerHu(torch.nn.Module):
#     def __init__(self, threshold=0.2):
#         super(BerHu, self).__init__()
#         self.threshold = threshold
#
#     def forward(pred, target):
#         mask = target>0
#         if not pred.shape == target.shape:
#             _,_,H,W = target.shape
#             pred = F.upsample(pred, size=(H,W), mode='bilinear')
#         pred = pred * mask
#         diff = torch.abs(target-pred)
#         delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]
#
#         part1 = -F.threshold(-diff, -delta, 0.)
#         part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
#         part2 = part2 / (2.*delta)
#
#         loss = part1 + part2
#         loss = torch.sum(loss)
#         return loss
