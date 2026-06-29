import os
import warnings
import argparse
import diffusers
import numpy as np
import torch
import random
from transformers import set_seed

warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = False
set_seed(42)

from utils.helpers import *
from utils.load_data import *

BACKBONE_DEFAULTS = {
    "moge2":    "guidance.moge2_guidance",
    "marigold": "guidance.marigold_guidance",
}

def main():
    parser = argparse.ArgumentParser(description="Poppy: Polarization-guided Normal Estimation")
    parser.add_argument("--backbone", type=str, choices=["marigold", "moge2"], required=True,
                        help="Backbone model to use")
    parser.add_argument("--mode", type=str, default=None,
                        help="Pipeline module to import (default: auto-selected from --backbone)")
    parser.add_argument("--data",    type=str, required=True,
                        help="Direct path to data directory (sfpuel format)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory for output images and log (default: <data>/output)")
    parser.add_argument("--gpu",      type=int,   default=0)
    parser.add_argument("--num_opt",  type=int,   default=100)
    parser.add_argument("--num_inf",  type=int,   default=4)
    parser.add_argument("--image_lr", type=float, default=0)
    parser.add_argument("--fs_lr",    type=float, default=0)
    parser.add_argument("--normal_lr",type=float, default=0)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data, "output")

    if args.mode is None:
        args.mode = BACKBONE_DEFAULTS[args.backbone]

    exec(f"from {args.mode} import *", globals())

    device = torch.device(f"cuda:{args.gpu}")

    if args.backbone == "marigold":
        from diffusers import DDIMScheduler
        pipe = marigold_guidance.from_pretrained(
            "prs-eth/marigold-normals-v1-1", prediction_type="normals"
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    else:
        pipe = moge2_guidance(device=device)

    vis_dir     = os.path.join(args.output, "vis")
    outputs_dir = os.path.join(args.output, "outputs")
    os.makedirs(vis_dir,     exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    f, csvwriter = open_log_file(args.output)

    for image_name in sorted(get_files(args.data)):
        with torch.no_grad():
            image, I_obs, Q_obs, U_obs, phi_obs, rho_obs, mask, normal_obs = \
                load_files(args.data, image_name, device)
            s_obs = torch.cat([I_obs, Q_obs, U_obs], dim=0)
            image = torch.clamp(image, 0, 1)

        outputs = pipe(
            image=image,
            s_obs=s_obs,
            phi_obs=phi_obs,
            rho_obs=rho_obs,
            normal_obs=normal_obs,
            mask=mask,
            num_inf=args.num_inf,
            num_opt=args.num_opt,
            seed=42,
            image_lr=args.image_lr,
            fs_lr=args.fs_lr,
            normal_lr=args.normal_lr,
        )

        pred = outputs[0] if isinstance(outputs, tuple) else outputs
        pred /= np.linalg.norm(pred, axis=2, keepdims=True)

        vis = normal_vis(pipe, pred, mask)
        if vis is not None:
            vis[0].save(os.path.join(vis_dir, image_name + ".png"))
        np.savez(os.path.join(outputs_dir, image_name + ".npz"), normals=pred)

        if normal_obs is not None:
            metrics = compute_metrics(
                pred,
                normal_obs.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32),
                mask,
            )
            mean_loss = float(metrics["mean"])
            median    = float(metrics["median"])
            rmse      = float(metrics["rmse"])
            acc_11    = float(metrics["acc_11"])
            acc_22    = float(metrics["acc_22"])
            acc_30    = float(metrics["acc_30"])

            print(
                f"[{image_name}] "
                f"mean={mean_loss:.4f} median={median:.4f} rmse={rmse:.4f} "
                f"acc11={acc_11:.4f} acc22={acc_22:.4f} acc30={acc_30:.4f}"
            )
            csvwriter.writerow([
                image_name,
                f"{mean_loss:.4f}", f"{median:.4f}", f"{rmse:.4f}",
                f"{acc_11:.4f}", f"{acc_22:.4f}", f"{acc_30:.4f}",
            ])
        else:
            print(f"[{image_name}] no ground-truth normals, skipping metrics")

    f.close()


if __name__ == "__main__":
    main()
