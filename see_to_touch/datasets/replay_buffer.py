import datetime
import io
import random
import traceback
from collections import defaultdict

from PIL import Image
from torchvision import transforms as T

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pickle

np.load.__defaults__ = (None, True, None, 'ASCII')
np.save.__defaults__ = (None, True, None, 'ASCII')

def episode_len(episode): 
    # subtract -1 because the dummy first transition
    return len(next(iter(episode.values()))) - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()
        # print('_DATA_SPECS: {}'.format(self._data_specs))

    def __len__(self):
        return self._num_transitions

    # Add value from _data_specs to _current_episode[spec.name] and
    # save it to a file
    def add(self, time_step, last=False):
        for spec in self._data_specs:
            if type(spec) is dict: # It means that this is observation
                value = time_step.observation
                for obs_spec in spec.values():
                    self._current_episode[obs_spec.name].append(value[obs_spec.name])
            else:
                value = time_step[spec.name]
                self._current_episode[spec.name].append(value)

        if last or time_step.last():
            episode = dict()
            for spec in self._data_specs:
                if type(spec) is dict:
                    for obs_spec in spec.values():
                        episode[obs_spec.name] = np.array(self._current_episode[obs_spec.name], obs_spec.dtype)
                else:
                    episode[spec.name] = np.array(self._current_episode[spec.name], spec.dtype)

            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)

class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, data_representations):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._data_representations = data_representations # We will return None for representations that are not going to be used
    
    #randomly sample an episode from the buffer
    def _sample_episode(self):
        # print('IN _SAMPLE_EPISODE EPISODE_FNS: {}'.format(self._episode_fns))
        # []
        eps_fn = random.choice(self._episode_fns)
        # print('EPS_FN in _sample_episode: {}'.format(eps_fn))
        return self._episodes[eps_fn]
    
    #fetch episodes from the storage and put in the buffer
    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)

        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
                # continue
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                continue

    def _sample(self):
        try:
            self._try_fetch() #fetch episodes from the storage and put in the buffer
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()

        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        if 'image' in self._data_representations:
            image_obs = episode['pixels'][idx-1] / 255.
            next_image_obs = episode['pixels'][idx+self._nstep-1] / 255.
        else:
            image_obs, next_image_obs = None, None

        if 'tactile' in self._data_representations:
            tactile_repr = episode['tactile'][idx-1]  
            next_tactile_repr = episode['tactile'][idx+self._nstep-1]
        else:
            tactile_repr, next_tactile_repr = None, None

        if 'allegro' in self._data_representations or 'kinova' in self._data_representations or 'franka' in self._data_representations:
            features = episode['features'][idx-1]
            next_features = episode['features'][idx+self._nstep-1]
        else:
            features, next_features = None, None

        action = episode['action'][idx]
        base_action = episode['base_action'][idx]
        base_next_action = episode['base_action'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        return (image_obs, tactile_repr, features, action, base_action, reward, discount, next_image_obs, next_tactile_repr, next_features, base_next_action)

    def __iter__(self):
        while True:
            # print('IN _ITER_ _EPISODES: {}'.format(self._episodes))
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, data_representations):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader




