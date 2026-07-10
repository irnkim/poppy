import builtins
import time
import torch
from moge.model.v2 import MoGeModel
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from utils.helpers import *

try:
    from typing import ParamSpec  # py>=3.10
except Exception:
    from typing_extensions import ParamSpec  # py3.9
builtins.ParamSpec = ParamSpec


class moge2_guidance:
    def __init__(self, ckpt="Ruicheng/moge-2-vitl-normal", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(ckpt).to(self.device).eval()

    def __call__(
        self,
        image, s_obs, rho_obs, mask,
        image_lr, fs_lr, normal_lr,
        num_opt=100,
        **_,
    ):
        self.image_processor = MarigoldImageProcessor(
            vae_scale_factor=8,
            do_normalize=True,
            do_range_check=True,
        )

        valid_mask = valid_mask_from_s0_both(s_obs[0:1], mask, rho_obs)

        param_groups = []
        if fs_lr >= 0:
            frac_s = torch.nn.Parameter(torch.ones_like(s_obs[0], device=self.device) * 1e-2)
            param_groups.append({"params": [frac_s], "lr": fs_lr})
        if normal_lr >= 0:
            normal_offset = torch.nn.Parameter(torch.ones_like(image, device=self.device) * 1e-2)
            param_groups.append({"params": [normal_offset], "lr": normal_lr})
        else:
            normal_offset = None
        if image_lr > 0:
            image_offset = torch.nn.Parameter(torch.ones_like(image) * 1e-2)
            param_groups.append({"params": [image_offset], "lr": image_lr})
        else:
            image_offset = None
        optimizer = torch.optim.Adam(param_groups) if param_groups else None

        min_tokens, max_tokens = 1200, 3600
        num_tokens = int(min_tokens + (9 / 9) * (max_tokens - min_tokens))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        for it in range(num_opt):
            if param_groups:
                optimizer.zero_grad(set_to_none=True)
            img_input = image + image_offset if image_offset is not None else image
            normal0 = self.model(img_input, num_tokens=num_tokens)['normal'].clone()
            normal0[..., 1] = -normal0[..., 1]  # flip y (down -> up)
            normal0[..., 2] = -normal0[..., 2]  # flip z (forward -> toward camera)
            zero = torch.linalg.norm(normal0, dim=-1) < eps
            normal0[zero] = torch.tensor([0.0, 0.0, 1.0], device=normal0.device)
            normal0 = normal0.permute(0, 3, 1, 2)
            normal0 = torch.nan_to_num(normal0, nan=0.0)
            if normal_lr > 0 and it > num_opt // 2:
                normal_hat = normal0 + normal_offset
            else:
                normal_hat = normal0

            Ls = frac_s
            Ld = s_obs[0] - Ls
            s_hat, _, _ = normal2polar_clip(s_obs[0], normal_hat, None, Ls, Ld)
            loss = polarization_physics_loss_Lopt(
                s_obs[:, valid_mask[0]], s_hat[:, valid_mask[0]],
            )
            if param_groups:
                loss.backward()
                optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - t_start

        with torch.no_grad():
            img_input = image + image_offset.detach() if image_offset is not None else image
            normal0 = self.model(img_input, num_tokens=num_tokens)['normal'].clone()
            normal0[..., 1] = -normal0[..., 1]
            normal0[..., 2] = -normal0[..., 2]
            zero = torch.linalg.norm(normal0, dim=-1) < eps
            normal0[zero] = torch.tensor([0.0, 0.0, 1.0], device=normal0.device)
            normal0 = normal0.permute(0, 3, 1, 2)
            normal0 = torch.nan_to_num(normal0, nan=0.0)
            if normal_lr > 0:
                normal0 = normal0 + normal_offset.detach()
            normal_hat = normal0 / (normal0.norm(dim=1, keepdim=True) + eps)

            Ls = frac_s.detach()
            Ld = (s_obs[0] - Ls).detach()

            return (
                normal_hat.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                Ls.permute(1, 2, 0).cpu().numpy(),
                Ld.permute(1, 2, 0).cpu().numpy(),
                elapsed_time,
            )
