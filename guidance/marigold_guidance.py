import warnings
import diffusers
import numpy as np
import torch
from diffusers import MarigoldNormalsPipeline

warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()
from utils.helpers import *

class marigold_guidance(MarigoldNormalsPipeline):
    def __call__(
        self,
        image, s_obs, rho_obs, mask,
        image_lr, fs_lr, normal_lr,
        num_inf=4, num_opt=25,
        processing_resolution=768, seed=42,
        **_,
    ) -> np.ndarray:
        device = self._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.no_grad():
            if self.empty_text_embedding is None:
                text_inputs = self.tokenizer(
                    "", padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]

        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution=processing_resolution, device=device, dtype=self.dtype
        )

        def latent_to_normal(latent) -> torch.Tensor:
            affine_invariant_prediction = self.decode_prediction(latent)
            normal = self.image_processor.unpad_image(affine_invariant_prediction, padding)
            return self.image_processor.resize_antialias(normal, original_resolution, "bilinear", is_aa=False)

        self.scheduler.set_timesteps(num_inf, device=device)

        valid_mask = valid_mask_from_s0_both(s_obs[0:1], mask, rho_obs)

        _, current_pred_latent = self.prepare_latents(image, None, generator, 1, 1)

        frac_s = torch.nn.Parameter(torch.ones_like(s_obs[0], device=device) * 1e-2)
        current_pred_latent = torch.nn.Parameter(current_pred_latent)
        normal_offset = torch.nn.Parameter(torch.ones_like(s_obs[0:1], device=device) * 1e-2)

        param_groups = []
        if fs_lr >= 0:
            param_groups.append({"params": [frac_s], "lr": fs_lr})
        if normal_lr > 0:
            param_groups.append({"params": [normal_offset], "lr": normal_lr})
        if image_lr > 0:
            image_offset = torch.nn.Parameter(torch.ones_like(image) * 1e-2)
            param_groups.append({"params": [image_offset], "lr": image_lr})
        else:
            image_offset = None
        optimizer = torch.optim.Adam(param_groups) if param_groups else None

        for its, t in enumerate(self.scheduler.timesteps):
            for it in range(num_opt):
                img_input = image + image_offset if image_offset is not None else image
                current_image_latent, _ = self.prepare_latents(img_input, None, generator, 1, 1)
                if param_groups:
                    optimizer.zero_grad()
                batch_latent = torch.cat([current_image_latent, current_pred_latent], dim=1)
                noise = self.unet(
                    batch_latent, t, encoder_hidden_states=self.empty_text_embedding, return_dict=False
                )[0]
                step_output = self.scheduler.step(noise, t, current_pred_latent, generator=generator)
                pred_original_sample = (
                    step_output.prev_sample if its == num_inf - 1
                    else step_output.pred_original_sample
                )
                normal = latent_to_normal(pred_original_sample)
                if its * num_opt + it > num_inf * num_opt // 2 and normal_lr > 0:
                    normal = normal + normal_offset
                Ls = frac_s
                Ld = s_obs[0] - Ls
                s_hat, _, _ = normal2polar_clip(s_obs[0], normal, None, Ls, Ld)
                loss = polarization_physics_loss_Lopt(
                    s_obs[:, valid_mask[0]], s_hat[:, valid_mask[0]],
                )
                if param_groups:
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                img_input = image + image_offset.detach() if image_offset is not None else image
                current_image_latent, _ = self.prepare_latents(img_input, None, generator, 1, 1)
                batch_latent = torch.cat([current_image_latent, current_pred_latent], dim=1)
                noise = self.unet(
                    batch_latent, t, encoder_hidden_states=self.empty_text_embedding, return_dict=False
                )[0]
                current_pred_latent.data = self.scheduler.step(
                    noise, t, current_pred_latent, generator=generator
                ).prev_sample

        with torch.no_grad():
            prediction = latent_to_normal(current_pred_latent.detach()) + normal_offset.detach()

        prediction = self.image_processor.pt_to_numpy(prediction)
        self.maybe_free_model_hooks()
        return prediction.squeeze()
