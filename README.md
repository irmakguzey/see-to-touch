# See to Touch: Learning Tactile Dexterity through Visual Incentives
[[Paper]](https://arxiv.org/abs/2309.12300) [[Project Website]](https://see-to-touch.github.io/) [[Data]](https://drive.google.com/drive/folders/1IpU97D4nSosdyyWmvuLO-E3phRA-nf8c?usp=sharing)

<p align="center">
  <img width="30%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/sponge_flipping.gif">
  <img width="30%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/eraser_turning.gif">
  <img width="30%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/plier_picking.gif">
 </p>

 <p align="center">
  <img width="30%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/mint_opening.gif">
  <img width="30%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/peg_insertion.gif">
  <img width="30%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/bowl_unstacking.gif">
</p>

**Authors**: [Irmak Guzey](https://irmakguzey.github.io/), [Yinlong Dai](https://www.linkedin.com/in/yinlong-dai-28aa35168/), [Ben Evans](https://bennevans.github.io/), [Soumith Chintala](https://soumith.ch/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University and Meta AI

This repository includes the official implementation of [TAVI](https://see-to-touch.github.io/). It includes online imitation learning (IL) algorithm used with implementations of offline IL, representation learning algorithms, RL agents, exploration and reward calculation modules. Our setup tackles 6 different dexterous manipulation tasks shown above and uses an [Allegro hand](https://www.wonikrobotics.com/research-robot-hand) with [XELA sensors integration](https://xelarobotics.com/en/integrations) and [Kinova arm](https://assistive.kinovarobotics.com/product/jaco-robotic-arm) as the hardware.

Demonstrations are collected through the [Holo-Dex](https://github.com/SridharPandian/Holo-Dex) pipeline and they are public in [this Google Drive link](https://drive.google.com/drive/folders/1IpU97D4nSosdyyWmvuLO-E3phRA-nf8c).

## Getting started
The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).

### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
  This will create a conda environment with the name `see_to_touch`. 
- Activate the environment:
  ```
  conda activate see_to_touch
  ```
- Install the `see_to_touch` package by using `setup.py`.
  ```
  pip install -e .
  ```
  This command should be done inside the conda environment.
  You can test if the project package has been installed correctly by running `import see_to_touch` from a python shell. 
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```

### Downloading the demonstration dataset
As mentioned the task demonstrations are public in [this Google Drive link](https://drive.google.com/drive/folders/1IpU97D4nSosdyyWmvuLO-E3phRA-nf8c). After compressing the `.zip` file 6 different demonstrations for each task can be reached
through:
```
<download-location>/see_to_touch/tasks/<bowl_unstacking|eraser_turning|plier_picking|sponge_flipping|peg_insertion|mint_opening>
```
Depending on the downloaded location, `data_path` variable in the config of task files should be updated. (Currently for each task it is
set to `/see_to_touch/tasks/<task-name>` as can be seen [here](https://github.com/irmakguzey/see-to-touch/blob/main/see_to_touch/configs/task/bowl_unstacking.yaml#L7)).

## Reproducing experiments

First you should activate the conda environment by running: `conda activate see_to_touch`. 

### Preprocessing
Before starting the training, each task data should be subsampled with respect to the total distance changed in the fingertips of the robot hand and the end effector position of the arm.

This can be done by running `python preprocess.py`. 
Necessary configurations should be made to the `see_to_touch/configs/preprocess.yaml` , such as the distances used for subsampling and which modalities (image, tactile, robot and etc.) to use in the system.

### SSL Pretraining
TAVI uses a new self-supervised representation learning algorithm that utilizes the temporal change and the object / robot configuration from the demonstrations.
It does this by combining an InfoNCE loss with positive pairs from nearby observations and the prediction loss of the state change between these observation.

These visual representations can be train by running:
```
python train.py encoder=temporal_ssl learner=temporal_ssl dataset=temporal_joint_diff_dataset data_dir=<task-data-path>
```

The model weights are saved under the hydra experiment created and should be used on the downstream online training.

We include the trainings of different visual representations mentioned in the paper as well. These can be trained by changing the `encoder`, `learner` and `dataset` variables of the `see_to_touch/configs/train.yaml` config file.

### Downloading the tactile encoder weights
TAVI uses tactile encoders trained in [T-Dex](https://tactile-dexterity.github.io/) from play data. This encoder weights are public [here](https://drive.google.com/drive/folders/148ycBmuvqkdESvAhStlhYJ7ISMgelEvL?usp=sharing). 
After downloading this folder, `tactile_encoder` folder path should be set as `tactile_out_dir` in `see_to_touch/configs/train_online.yaml` config file.

### Online Imitation Learning

TAVI uses DrQv2 as the main reinforcement learning (RL) agent and the rewards are calculated using the optimal-transport calculation between the last frame of the demonstration and the last 10 frames of the robot trajectory.

We have different implementations of each of the module in our setup.

After finishing the previous steps and setting the `image_out_dir` variable to the hydra experiment folder path from SSL pretraining for each task (for example look [here](https://github.com/irmakguzey/see-to-touch/blob/main/see_to_touch/configs/task/bowl_unstacking.yaml#L1)), you can start the online training as:

```
python train_online.py agent=tavi rl_learner=<drqv2|drq> base_policy=<openloop|vinn|vinn_openloop> explorer=ou_noise rewarder=<sinkhorn_cosine|sinkhorn_euclidean|cosine|euclidean> task=<task-name>
```

First value of each variable is as how it is in the publicized version of our implementation. We included the possible experimentations that can be done in this command as well.

Also, `episode_frame_matches` and `expert_frame_matches` can be modified to experiment further with the frames to include in reward calculation.


## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@misc{guzey2023touch,
    title={See to Touch: Learning Tactile Dexterity through Visual Incentives}, 
    author={Irmak Guzey and Yinlong Dai and Ben Evans and Soumith Chintala and Lerrel Pinto},
    year={2023},
    eprint={2309.12300},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}