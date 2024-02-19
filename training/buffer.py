import numpy as np


class DictReplayBuffer:
    def __init__(
        self,
        capacity,
        obs_dict_shape,
        act_shape,
        obs_dict_type=np.float32,
        act_type=np.float32,
    ):
        self.capacity = capacity
        self.observations = {}
        self.next_observations = {}
        for k in obs_dict_shape.keys():
            self.observations[k] = np.zeros(
                (self.capacity,) + obs_dict_shape[k],
                dtype=(
                    obs_dict_type[k] if type(obs_dict_type) is dict else obs_dict_type
                ),
            )
            self.next_observations[k] = np.zeros(
                (self.capacity,) + obs_dict_shape[k],
                dtype=(
                    obs_dict_type[k] if type(obs_dict_type) is dict else obs_dict_type
                ),
            )
        self.actions = np.zeros((self.capacity,) + act_shape, dtype=act_type)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.capacity
        return self.pos

    def push(self, obs, act, rew, next_obs, done):

        for k in self.observations.keys():
            self.observations[k][self.pos] = np.array(obs[k]).copy()
            self.next_observations[k][self.pos] = np.array(next_obs[k]).copy()
        self.actions[self.pos] = np.array(act).copy()
        self.rewards[self.pos] = np.array(rew).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.capacity:
            self.pos = 0
            self.full = True

    def push_batch(self, obs, act, rew, next_obs, done):
        k_tmp = list(obs.keys())[0]
        batch_size = len(obs[k_tmp])
        if batch_size >= self.capacity:
            # Replace the entire buffer
            for k in self.observations.keys():
                self.observations[k][:] = obs[k][-self.capacity :]
                self.next_observations[k][:] = next_obs[k][-self.capacity :]
            self.actions[:] = act[-self.capacity :]
            self.rewards[:] = rew[-self.capacity :]
            self.dones[:] = done[-self.capacity :]
            self.pos = 0
            self.full = True
        else:
            chunk = min(batch_size, self.capacity - self.pos)
            for k in self.observations.keys():
                self.observations[k][self.pos : self.pos + chunk] = obs[k][:chunk]
                self.next_observations[k][self.pos : self.pos + chunk] = next_obs[k][
                    :chunk
                ]
            self.actions[self.pos : self.pos + chunk] = act[:chunk]
            self.rewards[self.pos : self.pos + chunk] = rew[:chunk]
            self.dones[self.pos : self.pos + chunk] = done[:chunk]
            if chunk < self.capacity - self.pos:
                # Not reaching the end of buffer, self.full does not change
                self.pos += chunk
            else:
                # Reaching the end of buffer, start from beginning
                rem = batch_size - chunk
                for k in self.observations.keys():
                    self.observations[k][:rem] = obs[k][chunk:]
                    self.next_observations[k][:rem] = next_obs[k][chunk:]
                self.actions[:rem] = act[chunk:]
                self.rewards[:rem] = rew[chunk:]
                self.dones[:rem] = done[chunk:]
                self.pos = rem
                self.full = True

    def sample(self, batch_size, replace=True):
        batch_inds = np.random.choice(len(self), size=batch_size, replace=replace)
        return self._get_samples(batch_inds)

    def iterate(self, batch_size):
        random_inds = np.random.permutation(len(self))
        for i in range(0, len(self) - batch_size, batch_size):
            batch_inds = random_inds[i : i + batch_size]
            yield self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        obs = {}
        next_obs = {}
        for k in self.observations:
            obs[k] = self.observations[k][batch_inds]
            next_obs[k] = self.next_observations[k][batch_inds]
        act = self.actions[batch_inds]
        rew = self.rewards[batch_inds]
        done = self.dones[batch_inds]
        return obs, act, rew, next_obs, done

    def save(self, path):
        np.savez(path, **self.__dict__, allow_pickle=True)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        for key in self.__dict__.keys():
            setattr(self, key, data[key])
