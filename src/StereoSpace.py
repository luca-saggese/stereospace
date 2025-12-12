from os.path import join
from typing import Union, Optional, Tuple

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf, DictConfig
from jaxtyping import Float

import torch
from torch import Tensor
import torchvision.transforms as transforms

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .models import (
    UNet2DConditionModel,
    UNet3DConditionModel,
    ReferenceAttentionControl,
)
from .geometry import get_plucker_coordinates, normalize_K, get_default_intrinsics


class StereoSpace:
    def __init__(
        self, cfg: Union[dict, DictConfig], device: Optional[str] = "cuda:0"
    ) -> None:
        self.cfg = cfg
        self.model_path = join(
            self.cfg.data.pretrained_model_path, self.cfg.data.checkpoint_name
        )
        self.device = device
        self.configure()
        self.transform_pixels = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts image to Tensor
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

    def configure(self) -> None:
        if self.cfg.weight_dtype == "fp16":
            self.dtype = torch.float16
        elif self.cfg.weight_dtype == "fp32":
            self.dtype = torch.float32
        elif self.cfg.weight_dtype == "bf16":
            assert torch.cuda.is_available(), "CUDA required for bf16 inference."
            assert (
                torch.cuda.is_bf16_supported()
            ), "This GPU/stack doesn't support bf16."
            self.dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported weight dtype for this run: {self.cfg.weight_dtype}"
            )

        # Load models.
        self.load_models()

    @staticmethod
    def from_pretrained_2d(cls, unet_config, state_dict):
        unet_additional_kwargs = {
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "use_zero_convs": False,
        }

        unet_config["_class_name"] = cls.__name__
        unet_config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ]
        unet_config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ]
        unet_config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        model = cls.from_config(unet_config, **unet_additional_kwargs)

        # load the weights into the model
        m, u = model.load_state_dict(state_dict, strict=False)

        params = [
            p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()
        ]

        return model

    @staticmethod
    def fetch_state_dict(
        pretrained_model_name_or_path_or_dict: str,
        weight_name: str,
        subfolder: str | None = None,
    ):
        file_path = hf_hub_download(
            pretrained_model_name_or_path_or_dict, weight_name, subfolder=subfolder
        )
        state_dict = torch.load(file_path, weights_only=True)
        return state_dict

    def load_models(self) -> None:
        model_path = join(
            self.cfg.data.pretrained_model_path, self.cfg.data.checkpoint_name
        )

        # VAE.
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            subfolder="vae",
        ).to(self.device)

        # Image processor.
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()

        # Image encoder.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_path,
            subfolder="CLIP-ViT-H-14-laion2B-s32B-b79K",
        ).to(self.device, dtype=self.dtype)

        # Reference Unet.
        config = UNet2DConditionModel.load_config(model_path)
        self.reference_unet = UNet2DConditionModel.from_config(config).to(
            self.device, dtype=self.dtype
        )
        reference_sd = self.fetch_state_dict(model_path, "reference_unet.pth")
        self.reference_unet.load_state_dict(reference_sd)

        # Denoising Unet.
        denoising_sd = self.fetch_state_dict(model_path, "denoising_unet.pth")

        self.denoising_unet = self.from_pretrained_2d(
            UNet3DConditionModel, config, denoising_sd
        ).to(self.device, dtype=self.dtype)

        self.unet_in_channels = self.denoising_unet.config.in_channels

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)

    def encode_images(self, rgb: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        latents = self.vae.encode(rgb).latent_dist.mean  # rgb [-1, 1]
        latents = latents * 0.18215
        return latents

    def decode_latents(
        self, latents: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        latents = 1 / 0.18215 * latents
        rgb = []
        for frame_idx in range(latents.shape[0]):
            rgb.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        rgb = torch.cat(rgb)
        rgb = (rgb / 2 + 0.5).clamp(0, 1)
        return rgb.squeeze(2)

    def prepare_pluecker_embeds(
        self,
        baseline: Float[Tensor, "B 1"],
        H: int,
        W: int,
        intrinsics: Optional[Float[Tensor, "B 3 3"]] = None,
        intrinsics_tgt: Optional[Float[Tensor, "B 3 3"]] = None,
        F: int = 8,
    ):
        B = baseline.shape[0]
        src_list = []
        tgt_list = []
        extr_list = []
        intr_list = []

        for i in range(B):
            b_i = baseline[i]

            # anchor src camera at identity pose
            c2w = torch.eye(4, device=self.device).unsqueeze(0).repeat(2, 1, 1)
            # move target camera according to the baseline (left or right)
            c2w[0, 0, 3] = -0.5 * b_i
            c2w[1, 0, 3] = 0.5 * b_i

            w2c = torch.linalg.inv(c2w)

            if intrinsics is not None:
                K_i = intrinsics[i].unsqueeze(0)  # [1,3,3]

                K_norm = normalize_K(K_i, H, W)  # [1,3,3]

                if intrinsics_tgt is not None:
                    K_tgt_i = intrinsics_tgt[i].unsqueeze(0)
                    K_tgt_norm = normalize_K(K_tgt_i, H, W)  # [1,3,3]
                    Ks = torch.cat([K_norm, K_tgt_norm], dim=0)  # [2,3,3]
                else:
                    Ks = torch.cat([K_norm, K_norm], dim=0)  # [2,3,3]
            else:
                Ks = None
                K_i = get_default_intrinsics().to(self.device)
                K_i[:, 0, 0] *= W
                K_i[:, 1, 1] *= H
                K_i[:, 0, 2] *= W
                K_i[:, 1, 2] *= H

            Ks_i = torch.cat([K_i, K_i], dim=0)

            plucker = get_plucker_coordinates(
                extrinsics_src=w2c[0],
                extrinsics=w2c,
                intrinsics=Ks,
                target_size=(H // F, W // F),
            )

            src_list.append(plucker[0])
            tgt_list.append(plucker[1])
            extr_list.append(w2c)
            intr_list.append(Ks_i)

        src_pluecker = torch.stack(src_list, dim=0)  # [B,C,H//F,W//F]
        tgt_pluecker = torch.stack(tgt_list, dim=0)  # [B,C,H//F,W//F]
        extrinsics = torch.stack(extr_list, dim=0)  # [B,2,4,4]
        intrinsics = torch.stack(intr_list, dim=0)  # [B,2,3,3]

        return src_pluecker, tgt_pluecker, extrinsics, intrinsics

    def get_reference_controls(
        self, batch_size: int
    ) -> Tuple[ReferenceAttentionControl, ReferenceAttentionControl]:
        reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=True,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            feature_fusion_type="attention_full_sharing",
        )
        writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=True,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
            feature_fusion_type="attention_full_sharing",
        )

        return reader, writer

    def perform_nvs(
        self,
        src_images: Float[Tensor, "B C H W"],
        baseline: Float[Tensor, "B 1"],
        intrinsics: Optional[Float[Tensor, "B 3 3"]] = None,
        intrinsics_tgt: Optional[Float[Tensor, "B 3 3"]] = None,
        init_noise: Optional[Float[Tensor, "B C H//F W//F"]] = None,
        init_latents: Optional[Float[Tensor, "B C H//F W//F"]] = None,
        return_latents: bool = False,
    ):
        batch_size, _, H, W = src_images.shape

        src_images = src_images.to(self.device, dtype=self.dtype)
        baseline = baseline.to(self.device, dtype=self.dtype)
        if intrinsics is not None:
            intrinsics = intrinsics.to(self.device, dtype=self.dtype)
        if intrinsics_tgt is not None:
            intrinsics_tgt = intrinsics_tgt.to(self.device, dtype=self.dtype)

        images_for_clip = ((src_images + 1) * 0.5).clamp(0, 1)
        images_for_clip = images_for_clip.detach().to("cpu", dtype=torch.float32)

        clip_inputs = self.clip_image_processor(
            images=images_for_clip, return_tensors="pt", do_rescale=False
        )
        pixel_values = clip_inputs.pixel_values.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            clip_embed = self.image_encoder(pixel_values).image_embeds  # [B,768]

        image_prompt_embeds = clip_embed.unsqueeze(1).to(self.device, dtype=self.dtype)

        sched_kwargs = OmegaConf.to_container(self.cfg.noise_scheduler_kwargs)
        if self.cfg.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        val_scheduler = DDIMScheduler(**sched_kwargs)

        val_scheduler.set_timesteps(self.cfg.num_inference_steps, device=self.device)
        num_train_timesteps = val_scheduler.config.num_train_timesteps

        if init_noise is not None and init_latents is not None:
            latents = init_latents.to(self.device, dtype=self.dtype)
            noise = init_noise.to(self.device, dtype=self.dtype)
        else:
            latents = torch.randn(
                batch_size,
                4,
                self.cfg.data.train_height // self.vae_scale_factor,
                self.cfg.data.train_height // self.vae_scale_factor,
            ).to(self.device, dtype=self.dtype)

            noise = torch.randn_like(latents)

        initial_t = torch.tensor([num_train_timesteps - 1] * batch_size).to(
            self.device, dtype=torch.long
        )

        latents_noisy_start = val_scheduler.add_noise(latents, noise, initial_t)

        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        image_prompt_embeds = torch.cat(
            [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
        )

        with torch.no_grad():
            # Prepare ref image latents.
            ref_image_latents = self.encode_images(src_images)
            ref_image_latents = ref_image_latents.to(self.device, dtype=self.dtype)

        src_plucker, tgt_plucker, viewmats, Ks = self.prepare_pluecker_embeds(
            baseline=baseline,
            H=H,
            W=W,
            intrinsics=intrinsics,
            intrinsics_tgt=intrinsics_tgt,
        )
        src_plucker = src_plucker.to(self.device, dtype=self.dtype)
        tgt_plucker = tgt_plucker.to(self.device, dtype=self.dtype)

        reference_control_reader, reference_control_writer = (
            self.get_reference_controls(batch_size)
        )

        reference_input = torch.cat(
            (ref_image_latents, src_plucker.type_as(ref_image_latents)), dim=1
        )

        # Forward reference images.
        self.reference_unet(
            reference_input.repeat(2, 1, 1, 1),  # torch.Size([bs*2, 4, 64, 64])
            torch.zeros(batch_size * 2).to(reference_input),  # torch.Size([bs*2])
            encoder_hidden_states=image_prompt_embeds,  # torch.Size([bs*2, 1, 768])
            dense_emb=src_plucker.repeat(2, 1, 1, 1),
            return_dict=False,
        )

        # Update the denosing net with reference features.
        reference_control_reader.update(reference_control_writer)

        timesteps = val_scheduler.timesteps
        latents_noisy = latents_noisy_start
        for t in timesteps:

            # Prepare latents.
            latent_model_input = val_scheduler.scale_model_input(latents_noisy, t)

            latent_model_input = torch.cat(
                (latent_model_input, tgt_plucker.type_as(latent_model_input)), dim=1
            ).unsqueeze(2)
            latent_model_input = torch.cat([latent_model_input] * 2)

            # Denoise.
            noise_pred = self.denoising_unet(
                latent_model_input,  # torch.Size([bs*2, 4, 1, 64, 64])
                t,  # torch.Size([])
                encoder_hidden_states=image_prompt_embeds,  # torch.Size([bs*2, 1, 768])
                dense_emb=tgt_plucker.repeat(
                    2,
                    1,
                    1,
                    1,
                ),
                return_dict=False,
            )[
                0
            ]  # torch.Size([bs*2, 4, 1, 64, 64])
            noise_pred = noise_pred.squeeze(2)

            # CFG.
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # t -> t-1
            latents_noisy = val_scheduler.step(
                noise_pred, t, latents_noisy, return_dict=False
            )[0]

        # Noise disappears eventually
        latents_clean = latents_noisy

        reference_control_reader.clear()
        reference_control_writer.clear()

        if return_latents:
            return latents_clean

        synthesized = self.decode_latents(latents_clean)
        return synthesized
