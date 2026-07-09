from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn

from hdp_nuplan.utils.normalizer import StateNormalizer
from hdp_nuplan.utils.traj_kinematics import detached_integral


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    sde,

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    supervision_type: str = None,
    eps: float = 1e-3,
):
    ego_future, neighbors_future, neighbor_future_mask = futures
    B, T, _ = ego_future.shape
    ego_future_vel = torch.diff(
        torch.cat([torch.zeros_like(ego_future[:, :1, :], device=ego_future.device), ego_future], dim=-2),
        dim=-2
    )
    ego_future_vel[..., 2:] = ego_future[..., 2:]
    all_gt = norm(ego_future_vel)

    t = torch.rand(B, device=all_gt.device) * (1 - eps) + eps # [B,]
    z = torch.randn_like(all_gt, device=all_gt.device) # [B, T, 4]

    mean, std = sde.marginal_prob(all_gt, t)
    std = std.view(-1, *([1] * (len(all_gt.shape)-1)))

    xT = mean + std * z
    
    v = sde.transform("noise->v", z, t, xT)
    
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs) # [B, T, 4]
    model_pred = decoder_output["score"] # [B, T, 4]
    
    ##########################################################################
    # Transformation of model prediction and loss space
    # The model outputs *model_type* and is supervised with *supervision_type*
    ##########################################################################
    supervision_type = supervision_type if supervision_type is not None else model_type
    pred_pattern = f"{model_type}->{supervision_type}"
    score = sde.transform(pred_pattern, model_pred, t, xT)

    if supervision_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1) # to avoid exploding variance
    elif supervision_type == "x_start":
        dpm_loss = torch.sum((score - all_gt)**2, dim=-1)
    elif supervision_type == "noise":
        dpm_loss = torch.sum((score - z)**2, dim=-1)
    elif supervision_type == "v":
        dpm_loss = torch.sum((score - v)**2, dim=-1) 
    loss["ego_planning_loss"] = dpm_loss.mean()
    
    ##########################################################################
    #                              Hybrid Loss                               #
    # Integration performed in \tau_0 space
    ##########################################################################
    pred_v = sde.transform(f"{model_type}->x_start", model_pred, t, xT)
    pred_v = norm.inverse(pred_v)
    pred_x = detached_integral(pred_v[..., :2], detach_window_size=10)
    loss["ego_planning_hybrid_loss"] = torch.sum((pred_x - ego_future[..., :2])**2, dim=-1).mean()

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output
