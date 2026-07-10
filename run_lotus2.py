# Adapted from Lotus-2: https://github.com/EnVision-Research/Lotus-2
import argparse
import logging
import os
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.utils import convert_unet_state_dict_to_peft
from utils.helpers import *
from utils.load_data import *
from peft import LoraConfig, set_peft_model_state_dict
from torch import nn
from guidance.lotus2_guidance import lotus2_guidance

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("huggingface_hub not available. Model auto-download will not work.")

DEFAULT_REPO_NAME = "jingheya/Lotus-2"

CORE_PREDICTOR_FILENAME = {
    "depth":  "lotus-2_core_predictor_depth.safetensors",
    "normal": "lotus-2_core_predictor_normal.safetensors",
}
LCM_FILENAME = {
    "depth":  "lotus-2_lcm_depth.safetensors",
    "normal": "lotus-2_lcm_normal.safetensors",
}
DETAIL_SHARPENER_FILENAME = {
    "depth":  "lotus-2_detail_sharpener_depth.safetensors",
    "normal": "lotus-2_detail_sharpener_normal.safetensors",
}


def get_model_path(model_path, repo_id, filename):
    if model_path is not None:
        return model_path

    if not HF_AVAILABLE:
        raise ImportError(
            f"huggingface_hub is required for auto-downloading {filename}. "
            "Please install it with: pip install huggingface_hub"
        )

    logging.info(f"Downloading {filename} from {repo_id}")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    repo_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
    full_path = os.path.join(repo_path, filename)
    if not os.path.exists(full_path):
        for root, _, files in os.walk(repo_path):
            if filename in files:
                full_path = os.path.join(root, filename)
                break
        else:
            raise FileNotFoundError(f"Could not find {filename} in the downloaded repository")
    return full_path


class Local_Continuity_Module(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)


def load_lora_and_lcm_weights(
    transformer,
    core_predictor_model_path,
    lcm_model_path,
    detail_sharpener_model_path,
    task_name,
):
    lora_rank = 128 if task_name == "depth" else 256
    device = transformer.device
    weight_dtype = transformer.dtype

    target_lora_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
    ]

    core_predictor_model_path = get_model_path(
        core_predictor_model_path, DEFAULT_REPO_NAME, CORE_PREDICTOR_FILENAME[task_name]
    )
    lcm_model_path = get_model_path(
        lcm_model_path, DEFAULT_REPO_NAME, LCM_FILENAME[task_name]
    )
    detail_sharpener_model_path = get_model_path(
        detail_sharpener_model_path, DEFAULT_REPO_NAME, DETAIL_SHARPENER_FILENAME[task_name]
    )

    # core predictor lora
    core_lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        init_lora_weights="gaussian", target_modules=target_lora_modules,
    )
    transformer.add_adapter(core_lora_config, adapter_name="core_predictor")

    core_lora_state_dict = lotus2_guidance.lora_state_dict(core_predictor_model_path)
    core_state_dict = {
        k.replace("transformer.", ""): v
        for k, v in core_lora_state_dict.items()
        if k.startswith("transformer.")
    }
    core_state_dict = convert_unet_state_dict_to_peft(core_state_dict)
    incompatible_keys = set_peft_model_state_dict(
        transformer, core_state_dict, adapter_name="core_predictor"
    )
    if incompatible_keys is not None:
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logging.warning(f"Unexpected keys in core_predictor: {unexpected_keys}")

    for name, param in transformer.named_parameters():
        if "core_predictor" in name:
            param.requires_grad = False
    logging.info("Loaded core predictor weights.")

    # local continuity module
    local_continuity_module = Local_Continuity_Module(transformer.config.in_channels // 4)
    lcm_state_dict = torch.load(lcm_model_path, map_location="cpu", weights_only=True)
    local_continuity_module.load_state_dict(lcm_state_dict)
    local_continuity_module.requires_grad_(False)
    local_continuity_module.to(device=device, dtype=weight_dtype)
    logging.info("Loaded local continuity module weights.")

    # detail sharpener lora
    sharpener_lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        init_lora_weights="gaussian", target_modules=target_lora_modules,
    )
    transformer.add_adapter(sharpener_lora_config, adapter_name="detail_sharpener")

    sharpener_lora_state_dict = lotus2_guidance.lora_state_dict(detail_sharpener_model_path)
    sharpener_state_dict = {
        k.replace("transformer.", ""): v
        for k, v in sharpener_lora_state_dict.items()
        if k.startswith("transformer.")
    }
    sharpener_state_dict = convert_unet_state_dict_to_peft(sharpener_state_dict)
    incompatible_keys = set_peft_model_state_dict(
        transformer, sharpener_state_dict, adapter_name="detail_sharpener"
    )
    if incompatible_keys is not None:
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logging.warning(f"Unexpected keys in detail_sharpener: {unexpected_keys}")

    for name, param in transformer.named_parameters():
        if "detail_sharpener" in name:
            param.requires_grad = False
    logging.info("Loaded detail sharpener weights.")

    return transformer, local_continuity_module


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Poppy + Lotus-2")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--core_predictor_model_path", type=str, default=None)
    parser.add_argument("--lcm_model_path",             type=str, default=None)
    parser.add_argument("--detail_sharpener_model_path",type=str, default=None)
    parser.add_argument("--backbone", type=str, default="lotus2", choices=["lotus2"])
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--data",           type=str,   required=True,
                        help="Direct path to data directory (sfpuel format)")
    parser.add_argument("--output",         type=str,   default=None,
                        help="Output directory (default: <data>/output)")
    parser.add_argument("--gpu",            type=int,   default=0)
    parser.add_argument("--num_opt",        type=int,   default=100)
    parser.add_argument("--num_inf",        type=int,   default=10)
    parser.add_argument("--image_lr",       type=float, default=5e-4)
    parser.add_argument("--fs_lr",          type=float, default=1e-2)
    parser.add_argument("--normal_lr",      type=float, default=1e-3)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data, "output")

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)
    device = torch.device(f"cuda:{args.gpu}")

    # load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)

    # load lora and lcm weights
    transformer, local_continuity_module = load_lora_and_lcm_weights(
        transformer,
        args.core_predictor_model_path,
        args.lcm_model_path,
        args.detail_sharpener_model_path,
        "normal",
    )

    # build pipeline
    cls = lotus2_guidance
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", num_train_timesteps=10,
    )
    pipe = cls.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipe.local_continuity_module = local_continuity_module
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    vis_dir     = os.path.join(args.output, "vis")
    outputs_dir = os.path.join(args.output, "outputs")
    os.makedirs(vis_dir,     exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    f, csvwriter = open_log_file(args.output)

    for image_name in sorted(get_files(args.data)):
        with torch.no_grad():
            image, I_obs, Q_obs, U_obs, _, rho_obs, mask, normal_obs = \
                load_files(args.data, image_name, device)
            s_obs = torch.cat([I_obs, Q_obs, U_obs], dim=0)
            image = torch.clamp(image, 0, 1)

        image_ts = image * 2.0 - 1.0
        height, width = image_ts.shape[2:]
        max_edge = max(height, width)
        if max_edge > 1024:
            process_res = 1024
        elif max_edge < 512:
            process_res = 512
        else:
            process_res = None

        with torch.no_grad():
            result, Ls, Ld, elapsed_time = pipe(
                rgb_in=image_ts,
                prompt="",
                num_inf=args.num_inf,
                process_res=process_res,
                s_obs=s_obs,
                mask=mask,
                rho_obs=rho_obs,
                image_lr=args.image_lr,
                fs_lr=args.fs_lr,
                normal_lr=args.normal_lr,
                num_opt=args.num_opt,
            )

        # result.images is BCHW tensor; convert to HxWxC numpy and flip x to dataset convention
        pred = result.images.detach().squeeze(0).cpu().float().numpy().transpose(1, 2, 0)
        pred[:, :, 0] *= -1
        pred /= np.linalg.norm(pred, axis=2, keepdims=True) + 1e-8

        vis_result = normal_vis(None, pred, mask)
        if vis_result is not None:
            vis_result[0].save(os.path.join(vis_dir, image_name + ".png"))
        save_Ls_Ld_vis(vis_dir, image_name, Ls, Ld)
        np.savez(os.path.join(outputs_dir, image_name + ".npz"), normals=pred, Ls=Ls, Ld=Ld)

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
                f"time={elapsed_time:.4f}s "
                f"mean={mean_loss:.4f} median={median:.4f} rmse={rmse:.4f} "
                f"acc11={acc_11:.4f} acc22={acc_22:.4f} acc30={acc_30:.4f}"
            )
            csvwriter.writerow([
                image_name, f"{elapsed_time:.4f}",
                f"{mean_loss:.4f}", f"{median:.4f}", f"{rmse:.4f}",
                f"{acc_11:.4f}", f"{acc_22:.4f}", f"{acc_30:.4f}",
            ])
        else:
            print(f"[{image_name}] no ground-truth normals, skipping metrics (time={elapsed_time:.4f}s)")
            csvwriter.writerow([image_name, f"{elapsed_time:.4f}"])

    f.close()


if __name__ == "__main__":
    main()
