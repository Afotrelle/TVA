import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BiContrastiveLoss(nn.Module):
    def __init__(self, initial_temp=1.0, final_temp=0.07, total_batches=5):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = self.calculate_decay_rate()

    def forward(self, clean_feat, adv_feat, bicos, single=False, window_size=1):
        B, D, T = clean_feat.shape
        if bicos == 1:
            clean_feat = clean_feat.reshape(B * T, D)
            adv_feat = adv_feat.reshape(B * T, D)
            clean_feat = F.normalize(clean_feat, p=2, dim=1)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)

            cos_sim = torch.matmul(clean_feat, adv_feat.t()) / self.current_temp
            labels = torch.eye(cos_sim.size(0), device=cos_sim.device)

            if single:
                loss_clean2adv = 0.0
            else:
                loss_clean2adv = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
            loss_adv_2clean = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()
            return (loss_clean2adv + loss_adv_2clean) / 2

        if bicos == 2:
            clean_feat = clean_feat.reshape(B, D * T)
            adv_feat = adv_feat.reshape(B, D * T)
            clean_feat = F.normalize(clean_feat, p=2, dim=1)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)

            cos_sim = torch.matmul(clean_feat, adv_feat.t()) / self.current_temp
            labels = torch.eye(cos_sim.size(0), device=cos_sim.device)
            loss_clean2adv = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
            loss_adv_2clean = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()
            return (loss_clean2adv + loss_adv_2clean) / 2

        return torch.zeros((), device=clean_feat.device, dtype=clean_feat.dtype)

    def update_temperature(self, batch_count):
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)

    def calculate_decay_rate(self):
        return -np.log(self.final_temp / self.initial_temp) / self.total_batches


def _forward_features(model, x):
    """Support both generic classifier wrappers and TAD feature-backbone models."""
    if hasattr(model, "preprocess"):
        try:
            return model(model.preprocess(x))
        except TypeError:
            pass

    try:
        return model(x)
    except TypeError:
        if hasattr(model, "preprocess"):
            return model(model.preprocess(x))
        raise


def _feature_loss(clean_feat, adv_feat, loss_type='l2', bicos=1, flow=1, temp=0.01):
    if loss_type == 'l1':
        loss_fn = nn.L1Loss(reduction='mean')
    else:
        loss_fn = nn.MSELoss(reduction='mean')

    loss = loss_fn(adv_feat, clean_feat.detach())

    if bicos:
        criterion = BiContrastiveLoss(initial_temp=temp, final_temp=temp, total_batches=1)
        loss = loss + criterion(adv_feat, clean_feat.detach(), bicos, single=False)

    if flow:
        cos_sim = F.cosine_similarity(adv_feat[:, :, 1:], adv_feat[:, :, :-1])
        loss = loss + (1.0 - cos_sim).mean()

    return loss


def attack_PGD(
        norm,
        model,
        video_batch,
        k_ratio,
        labels_gt=None,
        alpha=0.1,
        lr_decay=0.8,
        num_iter=10,
        early_stop=False,
        targeted=False,
        target_class=None,
        budget_per_frame=True,
        frame_budget=None,
        original_preds=None,
        verbose=0,
        loss_type='l2',
        bicos=1,
        flow=1,
        temp=0.01):

    start = time.time()

    # If missing batch dimension (i.e. single video tensor), add it
    if len(video_batch.shape) == 4:
        video_batch = video_batch.unsqueeze(0)

    B, T, C, H, W = video_batch.shape

    # ---- Precompute original clean features for the TVA-style feature-space attack ----
    with torch.no_grad():
        clean_feat = _forward_features(model, video_batch)
        if isinstance(clean_feat, (tuple, list)):
            clean_feat = clean_feat[0]

    # ---- Initialize ----
    x_adv = video_batch.detach().clone()

    times = []

    # ---- Loop ----
    for i in range(num_iter):

        # ---- Gradient ----
        x_adv = x_adv.requires_grad_(True)


        adv_feat = _forward_features(model, x_adv)
        if isinstance(adv_feat, (tuple, list)):
            adv_feat = adv_feat[0]

        # TVA-style feature-space loss used throughout the repo/paper
        loss = _feature_loss(clean_feat, adv_feat, loss_type=loss_type, bicos=bicos, flow=flow, temp=temp)

        if targeted:
            if early_stop and (torch.argmax(adv_feat, dim=-1).eq(target_class).all()):
                break
        elif labels_gt is not None:
            if early_stop and (labels_gt != torch.argmax(adv_feat, dim=-1)).all():
                break

        # ---- Gradient ----
        grad = torch.autograd.grad(loss, x_adv)[0]

        # Gradient scaling
        grad = grad / grad.abs().mean().clamp(min=1e-8)

        with torch.no_grad():

            # ---- Update + Clamp ----
            x_adv = torch.clamp(
                x_adv + (lr_decay**i) * alpha * grad,
                0.0,
                1.0
            )
            # ---- Perturbation ----
            perturbation = x_adv - video_batch

            # ---- Optional temporal frame budget: select frames first ----
            if frame_budget is not None:
                if 0.0 <= frame_budget < 1.0:
                    num_frames = max(1, int(round(frame_budget * T)))
                else:
                    num_frames = max(1, int(frame_budget))
                num_frames = min(T, num_frames)
                frame_scores = perturbation.abs().sum(dim=(2, 3, 4))  # [B, T]
                _, topk_frame_idx = torch.topk(frame_scores, k=num_frames, dim=1)

                temp_mask = torch.zeros_like(frame_scores, dtype=torch.bool)
                for b in range(B):
                    temp_mask[b, topk_frame_idx[b]] = True

                frame_mask = temp_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                perturbation = perturbation * frame_mask.to(perturbation.dtype)

            # ---- Projection ----
            if budget_per_frame:
                # k_ratio budget independently for each selected frame
                k = int(k_ratio * C * H * W)

                perturbation_flat = perturbation.view(
                    B, T, -1
                )

            else:
                # k_ratio budget across the selected frames only
                if frame_budget is not None:
                    k = int(k_ratio * C * H * W * num_frames)
                else:
                    k = int(k_ratio * C * H * W * T)

                perturbation_flat = perturbation.view(
                    B, -1
                )

            if norm == 'l0':
                mask = torch.zeros_like(perturbation_flat)

                _, topk_indices = torch.topk(
                    perturbation_flat.abs(),
                    k,
                    dim=-1
                )

                mask.scatter_(
                    -1,
                    topk_indices,
                    1.0
                )

                perturbation = (
                    perturbation_flat * mask
                ).view(
                    B, T, C, H, W
                )
            
            elif norm == 'l1':
                perturbation_flat = l1_ball_projection_batch(
                    perturbation_flat,
                    k
                )

                perturbation = perturbation_flat.view(
                    B, T, C, H, W
                )

            # ---- Reconstruction ----
            x_adv = video_batch + perturbation

            # ---- Clamp ----
            x_adv = torch.clamp(
                x_adv,
                0.0,
                1.0
            )

            # Correct tiny numerical instabilities
            tol = 1e-6
            diff = x_adv - video_batch
            mask = torch.abs(diff) <= tol
            x_adv[mask] = video_batch[mask]

        # Break graph
        x_adv = x_adv.detach()

        # ---- Timing / logging ----
        times.append(time.time() - start)

        if verbose:
            elapsed = time.time() - start
            iter_per_sec = (i + 1) / elapsed
            remaining_iters = num_iter - (i + 1)
            eta = (
                remaining_iters / iter_per_sec
                if iter_per_sec > 0
                else 0
            )

            print(
                f'\rIteration {i+1}/{num_iter} | '
                f'Elapsed: {elapsed:.1f}s | '
                f'ETA: {eta:.1f}s'
                + ' ' * 5,
                end=''
            )

    if verbose:
        print(
            f'\rDone in {i+1} iterations! | '
            f'Total time: {time.time() - start:.1f}s'
            + ' ' * 3
        )

    return x_adv, preds, times


def l1_ball_projection_batch(x, tau):
    """
    x: shape (B, T, D) or generally (..., D)
    returns projected tensor of same shape
    """
    orig_shape = x.shape
    D = orig_shape[-1]

    x_flat = x.reshape(-1, D)          # (N, D), N = B*T
    abs_x = x_flat.abs()

    l1_norm = abs_x.sum(dim=1, keepdim=True)
    inside = l1_norm <= tau

    # Sort descending along feature dimension
    u, _ = torch.sort(abs_x, dim=1, descending=True)

    cssv = torch.cumsum(u, dim=1)

    arange = torch.arange(1, D + 1, device=x.device).view(1, -1)

    cond = u * arange > (cssv - tau)

    rho = cond.sum(dim=1) - 1          # (N,)

    theta = (
        cssv[torch.arange(x_flat.size(0), device=x.device), rho] - tau
    ) / (rho + 1).float()

    theta = theta.unsqueeze(1)

    w = torch.clamp(abs_x - theta, min=0.0)
    proj = torch.sign(x_flat) * w

    # restore vectors already inside the ball
    proj[inside.squeeze(1)] = x_flat[inside.squeeze(1)]

    return proj.reshape(orig_shape)