import torch

# SOURCE:
# https://github.com/xanderchf/MonoDepth-FPN-PyTorch/blob/master/main_fpn.py
class BerHu(torch.nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(pred, target):
        mask = target>0
        if not pred.shape == target.shape:
            _,_,H,W = target.shape
            pred = F.upsample(pred, size=(H,W), mode='bilinear')
        pred = pred * mask
        diff = torch.abs(target-pred)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()

    def forward(self, pred, target):
        if not pred.shape == target.shape:
            _,_,H,W = target.shape
            pred = F.upsample(pred, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(target)-torch.log(pred)) ** 2 ) )
        return loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, pred, target):
        if not pred.shape == target.shape:
            _,_,H,W = target.shape
            pred = F.upsample(pred, size=(H,W), mode='bilinear')
        loss = torch.mean( torch.abs(10.*target-10.*pred) )
        return loss

class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, pred, target):
        if not pred.shape == target.shape:
            _,_,H,W = target.shape
            pred = F.upsample(pred, size=(H,W), mode='bilinear')
        loss = torch.mean( torch.abs(torch.log(target)-torch.log(pred)) )
        return loss

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, pred, target):
        if not pred.shape == target.shape:
            _,_,H,W = target.shape
            pred = F.upsample(pred, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(10.*target-10.*pred) ** 2 ) )
        return loss

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_pred, grad_target):

        return torch.sum( torch.mean( torch.abs(grad_target-grad_pred) ) )

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_pred, grad_target):
        prod = ( grad_pred[:,:,None,:] @ grad_target[:,:,:,None] ).squeeze(-1).squeeze(-1)
        pred_norm = torch.sqrt( torch.sum( grad_pred**2, dim=-1 ) )
        target_norm = torch.sqrt( torch.sum( grad_target**2, dim=-1 ) )

        return 1 - torch.mean( prod/(pred_norm*target_norm) )
