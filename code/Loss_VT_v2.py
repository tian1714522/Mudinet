import torch
import torch.nn.functional as F


class MultiTaskVAELoss(torch.nn.Module):
    def __init__(self,beta):
        super(MultiTaskVAELoss, self).__init__()
        self.beta = beta

    def kl_divergence(self, mu, logvar):
        return torch.mean(0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1. - logvar, dim=1),dim=0)
        # return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    def multitime_kl(self,mu, logvar):
        return torch.mean(0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1. - logvar, dim=2))

    def forward(self, positions, gt_positions, recon_x, x, global1_mu, global1_logvar, global2_mu, global2_logvar, local_mu, local_logvar):
        # 重构损失：MSE
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL散度：全局变量1和标准正态分布
        kl_global1 = self.kl_divergence(global1_mu, global1_logvar)

        # KL散度：全局变量2和标准正态分布
        kl_global2 = self.kl_divergence(global2_mu, global2_logvar)

        # KL散度：局部变量和标准正态分布（对所有时刻求平均）
        kl_local = self.multitime_kl(local_mu, local_logvar)  # local_mu 和 local_logvar 应包含时间维度

        # 独立性KL散度：全局变量1和全局变量2之间的独立性项
        # kl_independence = torch.sum(global1_mu * global2_mu)

        # 定位精度损失

        position_loss = F.mse_loss(positions, gt_positions, reduction='mean')

        # 总损失
        total_loss = recon_loss + position_loss + self.beta[0]*kl_global1 + self.beta[1]*kl_global2 + self.beta[2]*kl_local# + kl_independence

        return total_loss,recon_loss,position_loss,kl_global1,kl_global2,kl_local
