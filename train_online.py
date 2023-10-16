# This script is used to train the policy online
import datetime
import os
import hydra

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dm_env import specs
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm 

from PIL import Image
# Custom imports 
from see_to_touch.datasets import *
from see_to_touch.models import *
from see_to_touch.utils import *


class Workspace:
    def __init__(self, cfg):
        # Set the variables
        self.work_dir = Path.cwd()
        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.data_path = cfg.data_path

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        # Run the setup - this should start the replay buffer and the environment
        self.data_path = cfg.data_path
        self.data_reprs = cfg.data_representations
        self._env_setup() # Should be set here

        # Set the image_transform
        self.image_episode_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        self._initialize_agent()

        self._global_step = 0 
        self._global_episode = 0

        # Set the logger right before the training
        self._set_logger(cfg)

    def _initialize_agent(self):
        action_spec = self.train_env.action_spec()
        action_shape = action_spec.shape
        print('action_shape: {}'.format(action_shape))

        print('self.cfg.agent: {}'.format(self.cfg.agent))
        self.agent = hydra.utils.instantiate(
            self.cfg.agent,
            action_shape = action_shape)
        self.agent.initialize_modules(
            rl_learner_cfg = self.cfg.rl_learner,
            base_policy_cfg = self.cfg.base_policy,
            rewarder_cfg = self.cfg.rewarder,
            explorer_cfg = self.cfg.explorer
        )

    def _set_logger(self, cfg):
        if self.cfg.log:
            wandb_exp_name = '-'.join(self.hydra_dir.split('/')[-2:])
            self.logger = Logger(cfg, wandb_exp_name, out_dir=self.hydra_dir)


    def _env_setup(self): # TODO: This should be parametrized

        self.train_env = hydra.utils.call( # If not call the actual interaction environment
            self.cfg.suite.task_make_fn
        )

        # Create replay buffer
        data_specs = [
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array(self.train_env.action_spec().shape, self.train_env.action_spec().dtype, 'base_action'),
            specs.Array((1,), np.float32, 'reward'), 
            specs.Array((1,), np.float32, 'discount')
        ]

        if self.cfg.buffer_path is None:
            replay_dir = self.work_dir / 'buffer' / self.cfg.experiment
        else:
            replay_dir = self.work_dir / 'buffer' / self.cfg.buffer_path
        
        self.replay_storage = ReplayBufferStorage(
            data_specs = data_specs,
            replay_dir = replay_dir # All the experiments are saved under same name
        )

        self.replay_loader = make_replay_loader(
            replay_dir = replay_dir,
            max_size = self.cfg.replay_buffer_size,
            batch_size = self.cfg.batch_size,
            num_workers = self.cfg.replay_buffer_num_workers,
            nstep = self.cfg.nstep,
            save_snapshot = self.cfg.suite.save_snapshot,
            discount = self.cfg.suite.discount,
            data_representations = self.cfg.data_representations
        )

        self._replay_iter = None
        if self.cfg.bc_regularize: # NOTE: If we use bc regularize you should create an expert replay buffer
            self.expert_replay_iter = None
        
        if self.cfg.evaluate:
            self.eval_video_recorder = TrainVideoRecorder( # It is the same recorder for our case
                save_dir = Path(self.work_dir) / 'online_training_outs/eval_video/videos' / self.cfg.experiment if self.cfg.save_eval_video else None,
                root_dir = None)
        self.train_video_recorder = TrainVideoRecorder(
            save_dir = Path(self.work_dir) / 'online_training_outs/train_video/videos' / self.cfg.experiment if self.cfg.save_train_video else None,
            root_dir = None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def save_snapshot(self, save_step=False, eval=False):
        snapshot = self.work_dir / 'weights'
        snapshot.mkdir(parents=True, exist_ok=True)
        if eval:
            snapshot = snapshot / ('snapshot_eval.pt' if not save_step else f'snapshot_{self.global_step}_eval.pt')
        else:
            snapshot = snapshot / ('snapshot.pt' if not save_step else f'snapshot_{self.global_step}.pt')
        keys_to_save = ['_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)
                
    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        # self.agent.load_snapshot(agent_payload) # NOTE: Make sure that this is okay
        self.agent.load_snapshot_eval(agent_payload)

    def _add_time_step(self, time_step, time_steps, observations):
        time_steps.append(time_step) # time_step is added directly

        if 'image' in self.data_reprs:
            pil_image_obs = Image.fromarray(np.transpose(time_step.observation['pixels'], (1,2,0)), 'RGB')
            transformed_image_obs = self.image_episode_transform(pil_image_obs)
            observations['image_obs'].append(transformed_image_obs)
        if 'tactile' in self.data_reprs:
            observations['tactile_repr'].append(torch.FloatTensor(time_step.observation['tactile']))
        if 'allegro' in self.data_reprs or 'kinova' in self.data_reprs or 'franka' in self.data_reprs:
            observations['features'].append(torch.FloatTensor(time_step.observation['features']))

        return time_steps, observations
    
    def _init_obs(self):
        obs = dict()
        if 'image' in self.data_reprs:
            obs['image_obs'] = list()
        if 'tactile' in self.data_reprs:
            obs['tactile_repr'] = list()
        if 'allegro' in self.data_reprs or 'kinova' in self.data_reprs or 'franka' in self.data_reprs:
            obs['features'] = list()

        return obs

    def _get_act_obs(self, time_step):
        obs_dict = dict()
        if 'image' in self.data_reprs:
            obs_dict['image_obs'] = torch.FloatTensor(time_step.observation['pixels'])
        if 'tactile' in self.data_reprs:
            obs_dict['tactile_repr'] = torch.FloatTensor(time_step.observations['tactile'])
        if 'allegro' in self.data_reprs or 'kinova' in self.data_reprs or 'franka' in self.data_reprs:
            obs_dict['features'] = torch.FloatTensor(time_step.observation['features'])

        return obs_dict

    def eval(self, evaluation_step):
        step, episode = 0, 0
        eval_until_episode = Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            episode_step = 0
            is_episode_done = False
            print(f"Eval Episode {episode}")
            time_steps = list() 
            observations = self._init_obs()

            time_step = self.train_env.reset()
            time_steps, observations = self._add_time_step(time_step, time_steps, observations)
            self.eval_video_recorder.init(time_step.observation['pixels'])

            while not (time_step.last() or is_episode_done):
                with torch.no_grad(), utils.eval_mode(self.agent):

                    action, base_action, is_episode_done, metrics = self.agent.act(
                        obs = self._get_act_obs(time_step),
                        global_step = self.global_step, 
                        episode_step = episode_step,
                        eval_mode = True # When set to true this will return the mean of the offsets learned from the model
                    )
                time_step = self.train_env.step(action, base_action)
                time_steps, observations = self._add_time_step(time_step, time_steps, observations)
                self.eval_video_recorder.record(time_step.observation['pixels'])
                step += 1
                episode_step += 1
                
                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_frame, 'global_frame')

            episode += 1
            x = input("Press Enter to continue... after reseting env")

            self.eval_video_recorder.save(f'{self.cfg.task.name}_eval_{evaluation_step}_{episode}.mp4')
        
        # Reset env
        self.train_env.reset()

    # Main online training code - this will be giving the rewards only for now
    def train_online(self):
        # Set the predicates for training
        train_until_step = Until(self.cfg.num_train_frames)
        seed_until_step = Until(self.cfg.num_seed_frames)
        eval_every_step = Every(self.cfg.eval_every_frames) # Evaluate in every these steps

        episode_step, episode_reward = 0, 0

        # Reset step implementations 
        time_steps = list() 
        observations = self._init_obs() 
        time_step = self.train_env.reset()
        
        self.episode_id = 0
        time_steps, observations = self._add_time_step(time_step, time_steps, observations)

        # if self.agent.auto_rew_scale:
        #     self.agent.sinkhorn_rew_scale = 1. # This will be set after the first episode

        self.train_video_recorder.init(time_step.observation['pixels'])
        metrics = dict() 
        is_episode_done = False
        while train_until_step(self.global_step): 
            
            # At the end of an episode actions
            if time_step.last() or is_episode_done:
                
                self._global_episode += 1 # Episode has been finished
                
                # Make each element in the observations to be a new array
                for obs_type in observations.keys():
                    observations[obs_type] = torch.stack(observations[obs_type], 0)

                # Get the rewards
                new_rewards = self.agent.get_reward( # NOTE: Observations is only used in the rewarder!
                    episode_obs = observations,
                    episode_id = self.global_episode,
                    visualize = self.cfg.save_train_cost_matrices
                )
                new_rewards_sum = np.sum(new_rewards)

                print(f'REWARD = {new_rewards_sum}')
                ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                self.train_video_recorder.save(f'{ts}_e{self.global_episode}_f{self.global_frame}_r{round(new_rewards_sum,2)}.mp4')
                
                # Update the reward in the timesteps accordingly
                obs_length = len(time_steps)
                for i, elt in enumerate(time_steps):
                    min_len = min(obs_length, self.cfg.episode_frame_matches) # Episode can be shorter than episode_frame_matches - NOTE: This looks liek a bug
                    if i > (obs_length - min_len):
                        new_reward = new_rewards[min_len - (obs_length - i)]
                        elt = elt._replace(reward=new_reward) # Update the reward of the object accordingly
                    self.replay_storage.add(elt, last = (i == len(time_steps) - 1))

                # Log
                if self.cfg.log:
                    metrics = {
                        'imitation_reward': new_rewards_sum,
                        'episode_reward': episode_reward
                    }
                    self.logger.log_metrics(metrics, time_step = self.global_episode, time_step_name = 'global_episode')

                # Reset the environment at the end of the episode
                time_steps = list()
                observations = self._init_obs()

                x = input("Press Enter to continue... after reseting env")

                time_step = self.train_env.reset()
                time_steps, observations = self._add_time_step(time_step, time_steps, observations)

                # Checkpoint saving and visualization
                self.train_video_recorder.init(time_step.observation['pixels'])
                if self.cfg.suite.save_snapshot:
                    self.save_snapshot()

                episode_step, episode_reward = 0, 0

            # Get the action
            with torch.no_grad(), eval_mode(self.agent):
                action, base_action, is_episode_done, metrics = self.agent.act(
                    obs = self._get_act_obs(time_step),
                    global_step = self.global_step, 
                    episode_step = episode_step,
                    eval_mode = False
                )
                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_frame, 'global_frame')

            print('STEP: {}'.format(self.global_step))
            print('---------')

            # Training - updating the agents 
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    replay_iter = self.replay_iter,
                    step = self.global_step
                )
                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_frame, 'global_frame')

            if self.cfg.evaluate and eval_every_step(self.cfg.eval_every_frames):
                self.eval(evaluation_step = int(self.global_step/self.cfg.eval_every_frames))
             
            # Take the environment steps    
            time_step = self.train_env.step(action, base_action)
            episode_reward += time_step.reward

            time_steps, observations = self._add_time_step(time_step, time_steps, observations)

            # Record and increase the steps
            self.train_video_recorder.record(time_step.observation['pixels']) # NOTE: Should we do env.render()? 
            episode_step += 1
            self._global_step += 1 

@hydra.main(version_base=None, config_path='see_to_touch/configs', config_name='train_online')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)

    if cfg.load_snapshot:
        snapshot = Path(cfg.snapshot_weight)
        print(f'Resuming the snapshot: {snapshot}')    
        workspace.load_snapshot(snapshot)

    workspace.train_online()


if __name__ == '__main__':
    main()