# God Mode Animation: 2D Game Animation Generation Model

<p float="left">
  <img src="/assets/godmodeanimation_logo.png?raw=true" width="300" />
  <img src="/assets/godmodeanimation_logo1.png?raw=true" width="300" /> 
</p>

I tried to train text to video and image to video models to generate 2D game animations. I used public game animation data and 3D mixamo model rendered animations to train the animation generation models. I'm open sourcing the model, training data, training code and data generation code. More details can be found in my blog [here](https://gavinliblog.com).

## text to animation, image to animation

<p align="middle">
  <img src="https://play.gptdash.xyz/demo_samples/050824/merged_spinkick_8.gif" width="200" />
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0009_0.gif" width="200" /> 
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_0.gif" width="200" /> 
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_7.gif" width="200" /> 
</p>

<p align="middle">
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0048_0.gif" width="200" />
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_2.gif" width="200" /> 
  <img src="https://play.gptdash.xyz/demo_samples/050824/merged_spinkick_0.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0052_2.gif" width="200" /> 
</p>

<p align="middle">
  <img src="https://play.gptdash.xyz/demo_samples/050824/merged_spinkick_6.gif" width="200" />
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_1.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0012_1.gif" width="200" /> 
  <img src="https://play.gptdash.xyz/demo_samples/050824/merged_spinkick_3.gif" width="200" /> 
</p>
<p align="middle">
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0058_2.gif" width="200" />
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0054_0.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0010_1.gif" width="200" /> 
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_3.gif" width="200" /> 
</p>
<p align="middle">
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_4.gif" width="200" />
  <img src="https://play.gptdash.xyz/demo_samples/050824/merged_spinkick_7.gif" width="200" /> 
  <img src="https://play.gptdash.xyz/demo_samples/050824/sword_6.gif" width="200" /> 
  <img src="https://f005.backblazeb2.com/file/godmodeaigendanims/samples/0014_2.gif" width="200" /> 
</p>

## text to game based on animation model

You can try the games [here](https://www.godmodeai.cloud/godmodedino/)
<p align="middle">
    <p align="middle">a dino jumping cacti in the desert</p>
    <p align="middle"><img src="/assets/dino_game.gif" width="300" /></p>
</p>
<p align="middle">
    <p align="middle">Donald Trump jumping trash can in new york city</p>
    <p align="middle"><img src="https://f005.backblazeb2.com/file/godmodeaigendanims/trump_game.gif" width="300" /></p>
</p>
<p align="middle">
    <p align="middle">Harry Potter jumping tree in Hogwarts castle</p>
    <p align="middle"><img src="https://f005.backblazeb2.com/file/godmodeaigendanims/harrypotter_game.gif?raw=true" width="300" /></p>
</p>
<p align="middle">
    <p align="middle">Tylor Swift jumping microphone in hotel room</p>
    <p align="middle"><img src="https://f005.backblazeb2.com/file/godmodeaigendanims/taylorswift_game.gif?raw=true" width="300" /></p>
</p>


### How to Use
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your-username/2D-game-animation.git
   ```
2. Install the necessary dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the datasets by placing them in the `/data` directory.
4. Run the main script to start generating animations.
   ```bash
   python scripts/generate_animation.py
   ```

### Requirements
- Python 3.8+
- TensorFlow / PyTorch
- OpenCV
- Unity / Unreal Engine (for game integration)

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### License
This project is licensed under the MIT License.

