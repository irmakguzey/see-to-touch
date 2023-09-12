# See to Touch: Learning Tactile Dexterity through Visual Incentives
[[Paper]](https://arxiv.org/abs/2303.12076) [[Project Website]](https://see-to-touch.github.io/) [[Data]](https://drive.google.com/drive/folders/1IpU97D4nSosdyyWmvuLO-E3phRA-nf8c?usp=sharing)

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

**Note:** Will be added after the publication.

<!-- Datasets for the play data and the demonstrations is uploaded in [this Google Drive link](https://drive.google.com/drive/folders/148ycBmuvqkdESvAhStlhYJ7ISMgelEvL). Instructions on how to use this dataset is given below.

## Getting started
The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).

### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
  This will create a conda environment with the name `tactile_dexterity`. 
- Activate the environment:
  ```
  conda activate tactile_dexterity
  ```
- Install the `tactile_dexterity` package by using `setup.py`.
  ```
  pip install -e .
  ```
  This command should be done inside the conda environment.
  You can test if the project package has been installed correctly by running `import tactile_dexterity` from a python shell. 
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```

### Pipeline Installation for Demonstration Collection
This work uses [Holo-Dex](https://github.com/SridharPandian/Holo-Dex) pipeline for demonstration collection and few of the interfaces implemented there. Also uses Holo-Dex package and API in order to connect to the robots.
You can install `holodex` as a separate package and follow the instructions to install pipeline and collect demonstrations. Same procedure is required for deployment as well. 

### Getting the tactile play data and the demonstration datasets
Datasets used for training and evaluation will be uploaded in [this Google Drive link](https://drive.google.com/drive/folders/148ycBmuvqkdESvAhStlhYJ7ISMgelEvL). 
- There will be two separate folders: 
    1. `play_data`: All the tactile play data. Kinova and Allegro commanded and current states are also saved along with tactile and visual observations.
    2. `evaluation`: Successful demonstrations used during robot runs. Each separate directory contains demonstrations for different tasks.
- Download and unzip the datasets. 
- Dataset paths will be updated in the configuration files. 

## Reproducing experiments
The following assumes our current working directory is the root folder of this project repository and the data provided above is being used. 
For each of these parts you should first activate the conda environment by running:
```
conda activate tactile_dexterity
```

### Preprocess
For both the play data trainings and the evaluation dataset we need a preprocessing procedure.
Steps for this are as follows:
1. Preprocess by running:
    ```
    python preprocess.py data_path=<data to preprocess>
    ```
    Here you should set the `data_path` variable to either the root task directory (which will be `<dataset_location>/evaluation/<task_name>`) or the play data directory (which will be `<dataset_location>/play_data`) .
    You can set the necessary parameters in `tactile_dexterity/configs/preprocess.yaml` file. 
2. Preprocessing should be done separately with different procedures.
    - If the preprocessing is done for tactile SSL training set the following parameters:
    ```
    vision_byol: false
    tactile_byol: true
    dump_images: false
    threshold_step_size: 0.1
    ```
    - If the preprocessing is done for image SSL training set the following parameters:
    ```
    vision_byol: true
    tactile_byol: false
    dump_images: true
    threshold_step_size: 0.1
    view_num: <camera-id>
    ```
    `view_num` parameter should be 0 if there is only one camera, otherwise it should be set to whichever camera you'd like to use.
    - If the preprocessing is done for robot deployments, then you should set the following parameters:
    ```
    vision_byol: true
    tactile_byol: false
    dump_images: true
    threshold_step_size: 0.2
    view_num: <camera-id>
    ```
    `threshold_step_size` can be changed according to the task but this is the default value. This parameter is the difference in the end effector position during subsampling. Please refer to the paper for more detailed information.

### Training
You can train encoders using Self-Supervised methods such as [BYOL](https://arxiv.org/abs/2006.07733) and [VICReg](https://arxiv.org/abs/2105.04906) for tactile and visual images. Also Behavior Cloning models with both image and the tactile. 
1. Use the following command to train a resnet encoder using the above mentioned SSL methods:
    ```
    python3 train.py data_dir=<path-to-desired-root> experiment=<experiment-name>
    ```
    Experiment name and the data directory to train can be changed accordingly. Experiment name is used for `wandb` especially.
    Dataroots used for preprocessing are the same for training.
2. Currently this command will train with BYOL with tactile images.In order to change the training type and the training object (tactile|image) one should modify `tactile_dexterity/configs/train.yaml` file. `learner`, `learner_type` and `dataset` variables should be changed accordingly. Namings are done similarly to the paper.

### Deploying Models
Trainings will save a snapshot of the used models under the `tactile-dexterity/out` with the time and experiment name. At each `deployer` model used the desired model paths should be retrieved from there.
You can deploy models by running:
```
python3 deploy.py data_path:<path-to-evaluation-task> deployer:<vinn|bc|openloop>
```
For each deployer module you should set the model directories to the snapshots saved. 
1. For VINN, set `tactile_out_dir` and `image_out_dir` to the desired encoders paths to be used.
2. For BC, set `out_dir` to the root of all the BC encoders that were saved during training.
3. For OpenLoop, you will not need an encoder.

**NOTE**: These instructions assume that you are running the `Holo-Dex` deployment API on a separate shell. Without communication to the robot these deployments cannot be done. -->

<!-- TODO: Add Citation when included to arxiv -->
<!-- ## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@misc{guzey2023dexterity,
  title={Dexterity from Touch: Self-Supervised Pre-Training of Tactile Representations with Robotic Play}, 
  author={Irmak Guzey and Ben Evans and Soumith Chintala and Lerrel Pinto},
  year={2023},
  eprint={2303.12076},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
} -->