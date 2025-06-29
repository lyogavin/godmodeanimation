# God Mode Animation: 2D Game Animation Generation Model

## 🎉 Updates @May 2025:

**We've launched [God Mode AI 2.0](https://www.godmodeai.co/ai-sprite-generator)** - AI sprite generator that can generate professional game animation sprites from your image, and much more!

<img src="/assets/godmodeai-hero-anim.webp" width="300" />

## Below is our last generation of model, for the latest work see above:

<p align="middle">
  <img src="/assets/godmodeanimation_logo.png?raw=true" width="300" />
  <img src="/assets/godmodeanimation_logo1.png?raw=true" width="300" /> 
</p>

I tried to train text to video and image to video models to generate 2D game animations. I'm open sourcing the model and training code.


[Website](https://godmodeai.co)

## text to animation, image to animation

<p align="middle">
  <img src="https://god-mode-ai.vercel.app/merged_spinkick_8.gif" width="200" />
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0009_0.gif" width="200" /> 
  <img src="https://god-mode-ai.vercel.app/sword_0.gif" width="200" /> 
  <img src="https://god-mode-ai.vercel.app/sword_7.gif" width="200" /> 
</p>

<p align="middle">
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0048_0.gif" width="200" />
  <img src="https://god-mode-ai.vercel.app/sword_2.gif" width="200" /> 
  <img src="https://god-mode-ai.vercel.app/merged_spinkick_0.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0052_2.gif" width="200" /> 
</p>

<p align="middle">
  <img src="https://god-mode-ai.vercel.app/merged_spinkick_6.gif" width="200" />
  <img src="https://god-mode-ai.vercel.app/sword_1.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0012_1.gif" width="200" /> 
  <img src="https://god-mode-ai.vercel.app/merged_spinkick_3.gif" width="200" /> 
</p>
<p align="middle">
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0058_2.gif" width="200" />
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0054_0.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0010_1.gif" width="200" /> 
  <img src="https://god-mode-ai.vercel.app/sword_3.gif" width="200" /> 
</p>
<p align="middle">
  <img src="https://god-mode-ai.vercel.app/sword_4.gif" width="200" />
  <img src="https://god-mode-ai.vercel.app/merged_spinkick_7.gif" width="200" /> 
  <img src="https://god-mode-ai.vercel.app/sword_6.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0014_2.gif" width="200" /> 
</p>

## text to game based on animation model

You can try the games [here](https://www.godmodeai.co/godmodedino/)
<p align="middle">
    <p align="middle">a dino jumping cacti in the desert</p>
    <p align="middle"><img src="/assets/dino_game.gif" width="300" /></p>
</p>
<p align="middle">
    <p align="middle">Donald Trump jumping trash can in new york city</p>
    <p align="middle"><img src="/assets/trump_game.gif" width="300" /></p>
</p>
<p align="middle">
    <p align="middle">Harry Potter jumping tree in Hogwarts castle</p>
    <p align="middle"><img src="/assets/harrypotter_game.gif" width="300" /></p>
</p>
<p align="middle">
    <p align="middle">Tylor Swift jumping microphone in hotel room</p>
    <p align="middle"><img src="/assets/taylorswift_game.gif" width="300" /></p>
</p>

## Trained Models


| Motion         | Epochs        | Steps        | Model Type              | HuggingFace Model         |
| ------------ | ------------- | ------------- | ------------------ | ------------ |
| Sword Wield | 36    |2035    | VC2 T2V | [model](https://huggingface.co/lyogavin/godmodeanimation_vc2_sword_wield_ep36)    |
| Sword Wield | 38    |2145    | VC2 T2V | [model](https://huggingface.co/lyogavin/godmodeanimation_vc2_sword_wield_ep38)    |
| Spin Kick | 32    |1815    | VC2 T2V | [model](https://huggingface.co/lyogavin/godmodeanimation_vc2_spinkick_ep32)    |
| Spin Kick | 34    |1925    | VC2 T2V | [model](https://huggingface.co/lyogavin/godmodeanimation_vc2_spinkick_ep34)    |
| Run Jump | 30    |3379    | VC2 T2V | [model](https://huggingface.co/lyogavin/godmodeanimation_vc2_runjump_ep30)    |
| Run Jump | 34    |3815    | VC2 T2V | [model](https://huggingface.co/lyogavin/godmodeanimation_vc2_runjump_ep34)    |
| Run | 19    |1080    | DC I2V | [model](https://huggingface.co/lyogavin/godmodeanimation_dc_run_ep19)    |




##  How to Train I2V Model

1. Clone the repository.
   ```bash
   git clone https://github.com/lyogavin/godmodeanimation.git
   ```
2. Install the necessary dependencies.
   ```bash
   cd godmodeanimation/i2v
   pip install -r requirements.txt
   ```
3. Prepare the datasets by unzipping the files and placing them in the `/root/ucf_ds` directory.
4. Download the pretrained model in the `/root/vc2/model.ckpt` directory.
5. Run the training script.
   ```bash
   train_t2v_run_jump.sh # for rum jump model
   train_t2v_spinkick.sh # for spin kick model
   train_t2v_sword_wield.sh # for sword wield model
   ```
## How to Train DC I2V Model

DC I2V model is based on [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter). Please follow the instructions in the [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) repository to train the DC I2V model.


## Replicate Public Model

We created Replicate public models for Sword Wield, Spin Kick, Run Jump and Run. You can try it from Replicat platform, find the details in [Replicate](https://replicate.com/lyogavin/godmodeanimation).


## Citing God Mode Animation

If you find
This work useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{godmodeanimation2024,
  author = {Gavin Li},
  title = {2D Game Animation Generation: All you need is repeat the same motion 1000 times},
  url = {https://github.com/lyogavin/godmodeanimation/},
  version = {1.0},
  year = {2024},
}
```


## Contribution 

Welcome contributions, ideas and discussions!

If you find this work useful or interesting to you, please ⭐ or buy me a coffee! It's very important to me! 🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)


