# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import os
import sys
import argparse
import random
from omegaconf import OmegaConf
from einops import rearrange, repeat
import torch
import torchvision
from pytorch_lightning import seed_everything
from cog import BasePredictor, Input, Path

sys.path.insert(0, "scripts/evaluation")
from funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    load_image_batch,
    get_filelist,
)
from utils.utils import instantiate_from_config


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""


        self.ckpt_path_runjump = "/runjump.ckpt"
        self.ckpt_path_spinkick = "/spinkick.ckpt"
        self.ckpt_path_swordwield = "/swordwield.ckpt"
        self.config_path = "configs/inference_t2v_512_v2.0.yaml"

        self.config_base = OmegaConf.load(self.config_path)
        self.model_config_base = self.config_base.pop("model", OmegaConf.create())
        self.model = None
        self.cached_task = None

    def predict(
        self,
        task: str = Input(
            description="Choose the task \"runjump\" for run and jump animation, \"spinkick\" for spin kick animation, \"swordwield\" for sword slashing animation.",
            choices=["runjump", "spinkick", "swordwield"],
            default="runjump",
        ),
        prompt: str = Input(
            description="Prompt for animtion, use this format: \"Game character [spin kick|run|jump|sword slashing] animation of [character description], side view, solid color background\"",
            default="Game character running animation of ancient Chinese swordsman, serene and wise, with flowing white robes and long beard, holding a slender, curved jian sword, side view, solid color background",
        ),
        ddim_steps: int = Input(description="Number of denoising steps.", default=50),
        unconditional_guidance_scale: float = Input(
            description="Classifier-free guidance scale.", default=12.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        save_fps: int = Input(
            description="Frame per second for the generated video.", default=10
        ),
    ) -> Path:

        width = 512
        height =  320
        # 
        if self.model is not None and self.cached_task != task:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

        if task == "runjump":
            self.model = instantiate_from_config(self.model_config_base)
            self.model = self.model.cuda()
            self.model = load_model_checkpoint(self.model, self.ckpt_path_runjump)
            self.model.eval()
            self.cached_task = "runjump"
        if task == "spinkick":
            self.model = instantiate_from_config(self.model_config_base)
            self.model = self.model.cuda()
            self.model = load_model_checkpoint(self.model, self.ckpt_path_spinkick)
            self.model.eval()
            self.cached_task = "spinkick"
        if task == "swordwield":
            self.model = instantiate_from_config(self.model_config_base)
            self.model = self.model.cuda()
            self.model = load_model_checkpoint(self.model, self.ckpt_path_swordwield)
            self.model.eval()
            self.cached_task = "swordwield"

        model = self.model

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        seed_everything(seed)

        args = argparse.Namespace(
            mode="base",
            savefps=save_fps,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=1.0,
            bs=1,
            height=height,
            width=width,
            frames=-1,
            fps=16,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_guidance_scale_temporal=None,
        )

        ## latent noise shape
        h, w = args.height // 8, args.width // 8
        frames = model.temporal_length if args.frames < 0 else args.frames
        channels = model.channels

        batch_size = 1
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps] * batch_size).to(model.device).long()
        prompts = [prompt]
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == "base":
            cond = {"c_crossattn": [text_emb], "fps": fps}
        else:
            raise NotImplementedError

        ## inference
        batch_samples = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            args.n_samples,
            args.ddim_steps,
            args.ddim_eta,
            args.unconditional_guidance_scale,
        )

        out_path = "/tmp/output.mp4"
        vid_tensor = batch_samples[0]
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(args.n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            out_path,
            grid,
            fps=args.savefps,
            video_codec="h264",
            options={"crf": "10"},
        )
        return Path(out_path)
