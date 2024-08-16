import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos_gif
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config

source_file_path = os.path.dirname(os.path.abspath(__file__)) # /a/b/c/d/e

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='/root/models/ep0025.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default=os.path.join(source_file_path, '../../configs/inference_t2v_512_v2.0.yaml'), help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    return parser


def run_inference(
        prompt, seed, uuid, save_path,
        **kwargs):

    start = time.time()


    args_str = f"--seed {seed} --savedir {save_path} "
    parser = get_parser()
    args = parser.parse_args(args_str.split())


    ## step 1. ensure model
    model, config = ensure_model(args)

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    ## saving folders
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    prompt_list = [prompt]
    num_samples = len(prompt_list)
    #filename_list = [f"u{uuid}_s{seed}"]


    ## step 3: run over samples
    ## -----------------------------------------------------------------
    #start = time.time()

    batch_size = 1
    #filenames = filename_list

    noise_shape = [batch_size, channels, frames, h, w]
    fps = torch.tensor([args.fps]*batch_size).to(model.device).long()


    #prompts = batch_size * [""]
    text_emb = model.get_learned_conditioning(prompt_list)

    cond = {"c_crossattn": [text_emb], "fps": fps}


    ## inference
    batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
                                            args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
    ## b,samples,c,t,h,w
    save_videos_gif(batch_samples, save_path, fps=args.savefps)

    print(f"Saved in {save_path}. Time used: {(time.time() - start):.2f} seconds")


    return save_path



model = None
config = None
def ensure_model(args):
    global model, config

    if model is not None:
        return model, config

    start = time.time()
    print(f"loading vc2 model {args.ckpt_path}...")

    assert os.path.exists(args.config)

    gpu_no = 0

    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    print(f"vc2 model {args.ckpt_path} finished laoding in {time.time() - start:.02f}sec.")



    return model, config




if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)