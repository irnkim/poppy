# Adapted from Lotus-2: https://github.com/EnVision-Research/Lotus-2
from typing import Union, Optional, List, Dict, Any
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils import is_torch_xla_available
from utils.helpers import *
from utils.image_utils import resize_image, resize_image_first

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class lotus2_guidance(FluxPipeline):
    def __call__(
        self,
        rgb_in: Optional[torch.FloatTensor] = None,
        prompt: Union[str, List[str]] = None,
        num_inf: int = 10,
        output_type: Optional[str] = "pil",
        process_res: Optional[int] = None,
        timestep_core_predictor: int = 1,
        guidance_scale: float = 3.5,
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        s_obs=None,
        mask=None,
        rho_obs=None,
        image_lr: float = 1e-2,
        fs_lr: float = 1e-2,
        normal_lr: float = 1e-4,
        num_opt: int = 0,
        **_,
    ):
        with torch.enable_grad():
            rgb_ori = rgb_in
            device = self._execution_device
            frac_s = torch.nn.Parameter(torch.ones_like(s_obs[0], device=device) * 1e-2)
            normal_offset = torch.nn.Parameter(torch.ones_like(s_obs[0:1], device=device) * 1e-2)

            param_groups = []
            if fs_lr >= 0:
                param_groups.append({"params": [frac_s], "lr": fs_lr})
            if image_lr > 0:
                image_offset = torch.nn.Parameter(torch.ones_like(rgb_ori) * 1e-2)
                param_groups.append({"params": [image_offset], "lr": image_lr})
            else:
                image_offset = None
            if normal_lr >= 0:
                param_groups.append({"params": [normal_offset], "lr": normal_lr})
            optimizer = torch.optim.Adam(param_groups) if param_groups else None

            valid_mask = valid_mask_from_s0_both(s_obs[0:1], mask, rho_obs)

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                    prompt=prompt, prompt_2=None, device=device,
                )

            for op in range(num_opt):
                if param_groups:
                    optimizer.zero_grad(set_to_none=True)

                rgb_input = rgb_ori + image_offset if image_offset is not None else rgb_ori
                batch_size = rgb_input.shape[0]
                input_size = rgb_input.shape[2:]
                rgb_in = resize_image_first(rgb_input, process_res)
                height, width = rgb_in.shape[2:]

                self._guidance_scale = guidance_scale
                self._joint_attention_kwargs = joint_attention_kwargs
                self._interrupt = False
                device = self._execution_device

                rgb_in = rgb_in.to(device=device, dtype=self.dtype)
                rgb_latents = self.vae.encode(rgb_in).latent_dist.sample()
                rgb_latents = (rgb_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                packed_rgb_latents = self._pack_latents(
                    rgb_latents,
                    batch_size=rgb_latents.shape[0],
                    num_channels_latents=rgb_latents.shape[1],
                    height=rgb_latents.shape[2],
                    width=rgb_latents.shape[3],
                )

                latent_image_ids_core_predictor = self._prepare_latent_image_ids(
                    batch_size, rgb_latents.shape[2] // 2, rgb_latents.shape[3] // 2, device, rgb_latents.dtype,
                )
                latent_image_ids = self._prepare_latent_image_ids(
                    batch_size, rgb_latents.shape[2] // 2, rgb_latents.shape[3] // 2, device, rgb_latents.dtype,
                )

                timestep_cp = (
                    torch.tensor(timestep_core_predictor)
                    .expand(batch_size)
                    .to(device=rgb_in.device, dtype=rgb_in.dtype)
                )

                sigmas = np.linspace(1.0, 1 / num_inf, num_inf)
                image_seq_len = packed_rgb_latents.shape[1]
                mu = calculate_shift(
                    image_seq_len,
                    self.scheduler.config.base_image_seq_len,
                    self.scheduler.config.max_image_seq_len,
                    self.scheduler.config.base_shift,
                    self.scheduler.config.max_shift,
                )
                timesteps, num_inf = retrieve_timesteps(
                    self.scheduler, num_inf, device, sigmas=sigmas, mu=mu,
                )
                num_warmup_steps = max(len(timesteps) - num_inf * self.scheduler.order, 0)
                self._num_timesteps = len(timesteps)

                if self.transformer.config.guidance_embeds:
                    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                    guidance = guidance.expand(packed_rgb_latents.shape[0])
                else:
                    guidance = None

                if self.joint_attention_kwargs is None:
                    self._joint_attention_kwargs = {}

                # core predictor
                self.transformer.set_adapter("core_predictor")
                latents = self.transformer(
                    hidden_states=packed_rgb_latents,
                    timestep=timestep_cp / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids_core_predictor,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = self.local_continuity_module(latents)

                # pack for detail sharpener, then decode directly (skip denoising loop during optimization)
                self.transformer.set_adapter("detail_sharpener")
                latents = self._pack_latents(
                    latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                latents_next = latents.to(dtype=self.dtype)
                latents_next = self._unpack_latents(latents_next, height, width, self.vae_scale_factor)
                latents_next = (latents_next / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents_next, return_dict=False)[0]
                normal_base = resize_image(image, input_size)
                normal_base[:, 0] = normal_base[:, 0] * -1

                if normal_lr > 0 and op > num_opt // 2:
                    normal_hat = normal_base + normal_offset
                else:
                    normal_hat = normal_base

                Ls = frac_s
                Ld = s_obs[0] - Ls
                s_hat, _, _ = normal2polar_clip(s_obs[0], normal_hat, fov=None, Ls=Ls, Ld=Ld)
                loss = polarization_physics_loss_Lopt(s_obs[:, valid_mask[0]], s_hat[:, valid_mask[0]])
                if param_groups:
                    loss.backward()
                    optimizer.step()

        # Final detail sharpener denoising loop (use detached offset for final pass)
        with torch.no_grad():
            rgb_input = rgb_ori + image_offset.detach() if image_offset is not None else rgb_ori
            rgb_in_final = resize_image_first(rgb_input, process_res)
            rgb_in_final = rgb_in_final.to(device=device, dtype=self.dtype)
            rgb_latents_final = self.vae.encode(rgb_in_final).latent_dist.sample()
            rgb_latents_final = (rgb_latents_final - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            packed_rgb_latents_final = self._pack_latents(
                rgb_latents_final,
                batch_size=rgb_latents_final.shape[0],
                num_channels_latents=rgb_latents_final.shape[1],
                height=rgb_latents_final.shape[2],
                width=rgb_latents_final.shape[3],
            )
            self.transformer.set_adapter("core_predictor")
            latents = self.transformer(
                hidden_states=packed_rgb_latents_final,
                timestep=timestep_cp / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids_core_predictor,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = self.local_continuity_module(latents)
            self.transformer.set_adapter("detail_sharpener")
            latents = self._pack_latents(
                latents,
                batch_size=latents.shape[0],
                num_channels_latents=latents.shape[1],
                height=latents.shape[2],
                width=latents.shape[3],
            )

        with self.progress_bar(total=num_inf) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = latents.to(dtype=self.dtype)
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = resize_image(image, input_size)
        if normal_lr > 0:
            image[:, 0] *= -1
            image = image + normal_offset.detach()
            image[:, 0] *= -1

        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)