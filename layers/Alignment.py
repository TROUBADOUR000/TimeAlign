import torch
import torch.nn as nn
import torch.nn.functional as F


class glocal_align_ablation(nn.Module):
    def __init__(self, local_margin=0.0, global_margin=0.0, loc=True, glo=True):
        super().__init__()
        self.local_margin = local_margin
        self.global_margin = global_margin
        self.loc = loc
        self.glo = glo

    def weight_based_dynamic_loss(self, losses):
        n = len(losses)
        w_avg = sum(loss.detach() for loss in losses) / n
        dyn_loss = sum(w_avg * loss / loss.detach() for loss in losses)
        
        return dyn_loss

    def forward(self, pred, target):
        # [B, C, T]
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        # Local point to point
        local_loss = torch.mean(F.gelu(1 - torch.abs(pred * target) - self.local_margin))
        # Align distribution
        global_loss = torch.mean(F.gelu(torch.abs(torch.matmul(pred, pred.transpose(1, 2)) - \
                                 torch.matmul(target, target.transpose(1, 2))) - self.global_margin))
        
        if not self.loc and not self.glo:
            return 0.0
        elif self.loc and not self.glo:
            return local_loss
        elif not self.loc and self.glo:
            return global_loss
        else:
            return self.weight_based_dynamic_loss([local_loss, global_loss])
            

class dual_align(nn.Module):
    def __init__(self, margin=0.05, tem=True, spa=True, glo=True):
        super().__init__()
        self.margin = margin
        self.tem = tem
        self.spa = spa
        self.glo = glo

    def weight_based_dynamic_loss(self, tem_loss, spa_loss, glo_loss):
        w_avg = (tem_loss.detach() + spa_loss.detach() + glo_loss.detach()) / 3.0

        w_tem = w_avg / tem_loss.detach()
        w_spa =  w_avg / spa_loss.detach()
        w_glo = w_avg / glo_loss.detach()
        
        return w_tem * tem_loss + w_spa * spa_loss + w_glo * glo_loss

    def forward(self, pred, target):
        # [B, C, T]
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        # Local
        temporal_loss = torch.mean(F.gelu(1.0 - self.margin - F.cosine_similarity(pred, target, dim=-1)))
        spatial_loss = torch.mean(F.gelu(1.0 - self.margin - F.cosine_similarity(pred, target, dim=-2)))
        # Global
        global_loss = torch.mean(F.gelu(torch.abs(torch.matmul(pred, pred.transpose(1, 2)) - \
                                 torch.matmul(target, target.transpose(1, 2))) - self.margin))

        return self.weight_based_dynamic_loss(temporal_loss, spatial_loss, global_loss)



class glocal_align(nn.Module):
    def __init__(self, local_margin=0.0, global_margin=0.0):
        super().__init__()
        self.local_margin = local_margin
        self.global_margin = global_margin

    def weight_based_dynamic_loss(self, losses):
        n = len(losses)
        w_avg = sum(loss.detach() for loss in losses) / n
        dyn_loss = sum(w_avg * loss / loss.detach() for loss in losses)
        
        return dyn_loss

    def forward(self, pred, target):
        # [B, C, T]
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        # Local point to point
        local_loss = torch.mean(F.gelu(1 - torch.abs(pred * target) - self.local_margin))
        # Align distribution
        global_loss = torch.mean(F.relu(torch.abs(torch.matmul(pred, pred.transpose(1, 2)) - \
                                 torch.matmul(target, target.transpose(1, 2))) - self.global_margin))

        return self.weight_based_dynamic_loss([local_loss, global_loss])



class orth_align(nn.Module):
    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def weight_based_dynamic_loss(self, losses):
        n = len(losses)
        w_avg = sum(loss.detach() for loss in losses) / n
        dyn_loss = sum(w_avg * loss / loss.detach() for loss in losses)
        
        return dyn_loss

    def forward(self, pred, target):
        # [B, C, T]
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        # gram
        gram_x = torch.matmul(pred, pred.transpose(1, 2))
        gram_y = torch.matmul(target, target.transpose(1, 2))
        # orthogonal
        diag_val_x = torch.diagonal(gram_x, dim1=-2, dim2=-1)
        diag_val_y = torch.diagonal(gram_y, dim1=-2, dim2=-1)
        diag_matrix_x = torch.diag_embed(diag_val_x)
        diag_matrix_y = torch.diag_embed(diag_val_y)  
        otrh_loss_x = torch.mean(F.gelu(torch.abs(gram_x - diag_matrix_x) - self.margin))
        otrh_loss_y = torch.mean(F.gelu(torch.abs(gram_y - diag_matrix_y) - self.margin))
        # Align
        align_loss = torch.mean(F.gelu(torch.abs(gram_x - gram_y) - self.margin))

        return self.weight_based_dynamic_loss([otrh_loss_x, otrh_loss_y, align_loss])



class Cosine_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # [B, C, D]
        # pred_n = nn.functional.normalize(pred, dim=-1)
        # target_n = nn.functional.normalize(target, dim=-1)
        # cos_loss = torch.mean((1 - F.cosine_similarity(pred_n, target_n, dim=-1)))
        cos_loss = torch.mean((1 - F.cosine_similarity(pred, target, dim=-1)))
        return cos_loss


class FreqSim_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_top_frequencies(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mag = torch.abs(xf)  # (B, F, C)
        k = mag.shape[1] // 2

        mag[:, 0, :] = 0

        top_mags, top_indices = torch.topk(mag, k, dim=1)  # (B, k, C)

        return top_indices, top_mags

    def forward(self, pred, target):
        pred_indices, pred_mags = self.get_top_frequencies(pred)
        target_indices, target_mags = self.get_top_frequencies(target)

        mag_loss = torch.mean(
            F.relu(1 - F.cosine_similarity(pred_mags, target_mags, dim=1))
        )

        return mag_loss


class KL_loss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, target):
        B, T, C = target.shape

        pred_perm = pred.permute(0, 2, 1)  # (B, C, T)
        target_perm = target.permute(0, 2, 1)  # (B, C, T)

        pred_probs = F.softmax(pred_perm / self.temperature, dim=-1)
        target_probs = F.softmax(target_perm / self.temperature, dim=-1)

        kld_loss = (
            1 / C * F.kl_div(
                pred_probs.log(), target_probs, reduction="batchmean", log_target=False
            )
        )

        return kld_loss


class AutoWeighted_Loss(nn.Module):
    def __init__(self, temperature=1.0, init_sigma=0.0):
        super().__init__()
        self.log_sigma_kld = nn.Parameter(torch.tensor(init_sigma))
        self.log_sigma_mse = nn.Parameter(torch.tensor(init_sigma))
        self.log_sigma_cos = nn.Parameter(torch.tensor(init_sigma))
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        B, T, C = target.shape
        # print(B, ' ', T, ' ', C)

        mse_loss = self.mse_loss(pred, target)
        cos_loss = torch.mean(F.relu(1 - F.cosine_similarity(pred, target, dim=1)))

        pred_perm = pred.permute(0, 2, 1)  # (B, C, T)
        target_perm = target.permute(0, 2, 1)  # (B, C, T)

        pred_probs = F.softmax(pred_perm / self.temperature, dim=-1)
        target_probs = F.softmax(target_perm / self.temperature, dim=-1)

        kld_loss = (
            1
            / C
            * F.kl_div(
                pred_probs.log(), target_probs, reduction="batchmean", log_target=False
            )
        )
        # print(mse_loss, ' ', cos_loss, ' ', kld_loss)
        # input()

        w_kld = 1.0 / (2.0 * torch.exp(self.log_sigma_kld))
        w_mse = 1.0 / (2.0 * torch.exp(self.log_sigma_mse))
        w_cos = 1.0 / (2.0 * torch.exp(self.log_sigma_cos))

        reg_term = 0.5 * (self.log_sigma_kld + self.log_sigma_mse + self.log_sigma_cos)

        total_loss = w_kld * kld_loss + w_mse * mse_loss + w_cos * cos_loss + reg_term

        return total_loss
