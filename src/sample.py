import os
from typing import Any
import math

import hydra
import torch
from pytorch_lightning.lite import LightningLite

import numpy as np
import omegaconf
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from torchvision.utils import save_image, make_grid

from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from utils.helpers import ensure_path_join, denormalize_to_zero_to_one

import sys
sys.path.insert(0, 'IDiff-Face/')

import torchvision.transforms as T
from PIL import Image


class DiffusionSamplerLite(LightningLite):
    def run(self, cfg) -> Any:

        # Load training configuration
        train_cfg_path = os.path.join(cfg.checkpoint.path, '.hydra', 'config.yaml')
        train_cfg = omegaconf.OmegaConf.load(train_cfg_path)

        # Seed for reproducibility
        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))

        # Instantiate and setup diffusion model
        diffusion_model = instantiate(train_cfg.diffusion)
        diffusion_model = self.setup(diffusion_model)

        # Load checkpoints
        if cfg.checkpoint.global_step is not None:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints',
                                         f'ema_averaged_model_{cfg.checkpoint.global_step}.ckpt')
        elif cfg.checkpoint.use_non_ema:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', 'model.ckpt')
        else:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', 'ema_averaged_model.ckpt')

        diffusion_model.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint_path}")

        # Determine sample size
        size = (train_cfg.constants.input_channels,
                train_cfg.constants.image_size,
                train_cfg.constants.image_size)


        # Setup VQGAN for latent diffusion if needed
        if train_cfg.latent_diffusion:
            latent_encoder = VQEncoderInterface(
                first_stage_config_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_config.yaml"),
                encoder_state_dict_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_encoder_state_dict.pt")
            )
            latent_encoder = self.setup(latent_encoder)
            latent_encoder.eval()


            # Update size to latent dimensions
            size = latent_encoder(torch.ones([1, *size]).to(self.device)).shape[-3:]

            latent_decoder = VQDecoderInterface(
                first_stage_config_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_config.yaml"),
                decoder_state_dict_path=os.path.join(".", "neg_prompt", "models", "autoencoder",
                                                     "first_stage_decoder_state_dict.pt")
            )
            latent_decoder = self.setup(latent_decoder)
            latent_decoder.eval()
        else:
            latent_decoder = None

        # Load contexts
        contexts_file = ensure_path_join(cfg.create_contexts.contexts_save_path,
                                       cfg.create_contexts.contexts_save_name +
                                       str(cfg.create_contexts.n_contexts) + ".npy")
        input_contexts_name = contexts_file.split("/")[-1].split(".")[0]
        model_name = cfg.checkpoint.path.split("/")[-1]

        contexts = torch.load(contexts_file, weights_only=False)
        contexts = {str(int(k)): v for k, v in contexts.items()}

        assert len(contexts) >= cfg.sampling.n_contexts
        context_ids = list(contexts.keys())[:cfg.sampling.n_contexts]


        # Prepare output directory path
        if cfg.neg_prompt.is_adaptive_w:
            if cfg.neg_prompt.is_reverse_adaptive:
                w_path = f"reverse_adaptive/w_{int(10 * cfg.neg_prompt.w)}"
            else:
                w_path = f"adaptive/w_{int(10 * cfg.neg_prompt.w)}"
        else:
            w_path = f"w_{int(10 * cfg.neg_prompt.w)}"

        if cfg.sampling.is_ddim:
            samples_dir = ensure_path_join(cfg.sampling.root_dir, model_name,
                                         input_contexts_name, cfg.sampling.method, w_path, "samples")
        else:
            samples_dir = ensure_path_join(cfg.sampling.root_dir, model_name,
                                         input_contexts_name, f"ddpm_{cfg.sampling.method}", w_path, "samples")

        # Skip already generated samples
        length_before = len(context_ids)
        context_ids = [i for i in context_ids if not os.path.isfile(os.path.join(samples_dir, f"{i}.png"))]
        print(f"[INFO] Skipped {length_before - len(context_ids)} already processed samples.")
        context_ids = self.split_across_devices(context_ids)

        if self.global_rank == 0:
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)

        # Main loop over contexts
        for id_name in context_ids:
            prefix = id_name

            # Positive context (P+ = Morph embedding)
            base_context = torch.from_numpy(contexts[id_name]).unsqueeze(0).to(self.device)
            context = base_context.repeat(cfg.sampling.batch_size, 1)

            # DDIM Inversion: Get x_T from morphed image
            morph_path_file = '/content/drive/MyDrive/HCML_Project/NegFaceDiff/current_morph_path.txt'
            if os.path.exists(morph_path_file):
                with open(morph_path_file, 'r') as f:
                    img_path = f.read().strip()
            else:
                # Fallback (should not happen if you ran the context script)
                img_path = '/content/drive/MyDrive/HCML_Project/MAD22_Data/extracted_images/original_sorted/OpenCV/030_08-vs-039_08.jpg'

            print(f"[SYSTEM] Preparing initial noise (x_T) from: {img_path}")

            transform = T.Compose([
                T.Resize((cfg.align.image_size, cfg.align.image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            img_pil = Image.open(img_path).convert('RGB')
            x_0_pixel = transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if train_cfg.latent_diffusion:
                    x_0_latent = latent_encoder(x_0_pixel)
                else:
                    x_0_latent = x_0_pixel

                base_x_T_latent = diffusion_model.module.ddim_inversion(
                    x_0_latent, context=base_context, ddim_step=cfg.sampling.ddim_step
                )

                x_T_latent = base_x_T_latent.repeat(cfg.sampling.batch_size, 1, 1, 1)
                context = base_context.repeat(cfg.sampling.batch_size, 1)

            # Negative context (P-)
            if cfg.neg_prompt.w != 0:
                neg_id = str(int(id_name) - 1) if int(id_name) > 0 else str(len(contexts) - 1)
                negative_context = torch.from_numpy(contexts[neg_id]).unsqueeze(0)
                negative_context = negative_context.repeat(cfg.sampling.batch_size, 1).to(self.device)

                self.perform_sampling(
                    diffusion_model=diffusion_model,
                    n_samples=cfg.sampling.n_samples_per_context,
                    size=size,
                    batch_size=cfg.sampling.batch_size,
                    samples_dir=samples_dir,
                    prefix=prefix,
                    context=context,
                    negative_context=negative_context,
                    w=cfg.neg_prompt.w,
                    is_adaptive_w=cfg.neg_prompt.is_adaptive_w,
                    is_reverse_adaptive=cfg.neg_prompt.is_reverse_adaptive,
                    ddim_step=cfg.sampling.ddim_step,
                    is_ddim=cfg.sampling.is_ddim,
                    latent_decoder=latent_decoder,
                    x_T=x_T_latent
                )
            else:
                self.perform_original_sampling(
                    diffusion_model=diffusion_model,
                    n_samples=cfg.sampling.n_samples_per_context,
                    size=size,
                    batch_size=cfg.sampling.batch_size,
                    samples_dir=samples_dir,
                    prefix=prefix,
                    context=context,
                    ddim_step=cfg.sampling.ddim_step,
                    is_ddim=cfg.sampling.is_ddim,
                    latent_decoder=latent_decoder,
                    x_T=x_T_latent
                )

    @staticmethod
    def perform_original_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None, ddim_step: int = 200,
            is_ddim: bool = True, latent_decoder: torch.nn.Module = None,
            x_T: torch.Tensor = None):

        n_batches = math.ceil(n_samples / batch_size)
        samples_for_grid = []

        with torch.no_grad():
            for _ in range(n_batches):
                if is_ddim:
                    batch_samples = diffusion_model.original_sample_ddim(
                        batch_size, size, x_T=x_T, context=context, ddim_step=ddim_step)
                else:
                    batch_samples = diffusion_model.original_sample_ddpm(
                        batch_size, size, context=context)

                if latent_decoder:
                    batch_samples = latent_decoder(batch_samples).cpu()
                    print(f"Decoded raw - min: {batch_samples.min().item():.4f}, max: {batch_samples.max().item():.4f}, mean: {batch_samples.mean().item():.4f}")
#                    print(f"Decoded shape: {batch_samples.shape}, min: {batch_samples.min()}, max: {batch_samples.max()}")

#                batch_samples = denormalize_to_zero_to_one(batch_samples)
                batch_samples = batch_samples.clamp(-1.0, 1.0)
                batch_samples = denormalize_to_zero_to_one(batch_samples)
                samples_for_grid.append(batch_samples)

            samples = torch.cat(samples_for_grid, dim=0)[:n_samples]
            grid = make_grid(samples, nrow=4, padding=0)
            save_image(grid, ensure_path_join(samples_dir, f"{prefix}.png"))

    @staticmethod
    def perform_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None,
            negative_context: torch.Tensor = None, w: float = 0.5,
            is_adaptive_w: bool = False, is_reverse_adaptive: bool = False,
            ddim_step: int = 200, is_ddim: bool = True,
            latent_decoder: torch.nn.Module = None, x_T: torch.Tensor = None):

        n_batches = math.ceil(n_samples / batch_size)
        samples_for_grid = []

        with torch.no_grad():
            for _ in range(n_batches):
                if is_ddim:
                    batch_samples = diffusion_model.sample_ddim(
                        batch_size, size, x_T=x_T, context=context,
                        negative_context=negative_context, w=w,
                        is_adaptive_w=is_adaptive_w,
                        is_reverse_adaptive=is_reverse_adaptive,
                        ddim_step=ddim_step)
                else:
                    batch_samples = diffusion_model.sample_ddpm(
                        batch_size, size, x_T=x_T, context=context,
                        negative_context=negative_context, w=w,
                        is_adaptive_w=is_adaptive_w,
                        is_reverse_adaptive=is_reverse_adaptive)

                if latent_decoder:
                    batch_samples = latent_decoder(batch_samples).cpu()

                    print(f"Decoded raw - min: {batch_samples.min().item():.4f}, max: {batch_samples.max().item():.4f}, mean: {batch_samples.mean().item():.4f}")


                #batch_samples = denormalize_to_zero_to_one(batch_samples)
                batch_samples = denormalize_to_zero_to_one(batch_samples.clamp(-1.0, 1.0))
                samples_for_grid.append(batch_samples)

            samples = torch.cat(samples_for_grid, dim=0)[:n_samples]
            grid = make_grid(samples, nrow=4, padding=0)
            save_image(grid, ensure_path_join(samples_dir, f"{prefix}.png"))

    def split_across_devices(self, L):
        if isinstance(L, int):
            L = list(range(L))
        if len(L) == 0:
            return []
        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [L[idx:idx + chunk_size] for idx in range(0, len(L), chunk_size)]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])
        return L_per_device[self.global_rank]

    @staticmethod
    def spherical_interpolation(value, start, target):
        start = torch.nn.functional.normalize(start)
        target = torch.nn.functional.normalize(target)
        omega = torch.acos((start * target).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - value) * omega) / so).unsqueeze(1) * start + \
              (torch.sin(value * omega) / so).unsqueeze(1) * target
        return res


@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":
    sample()