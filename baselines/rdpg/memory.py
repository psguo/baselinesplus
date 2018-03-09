import numpy as np
from collections import deque
import copy

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, horizonlen, action_shape, observation_shape):
        self.limit = limit
        self.horizonlen = horizonlen
        self.currentMemEps = MemoryEps(horizonlen, action_shape, observation_shape)
        self.history = deque(maxlen=limit)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.observations0.shape))
        obs1_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.observations1.shape))
        action_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.actions.shape))
        reward_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.rewards.shape))
        terminal1_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.terminals1.shape))

        for i in batch_idxs:
            obs0_batch[i] = self.history[i].get_data()
            obs1_batch[i] = self.history[i].get_data()
            action_batch[i] = self.history[i].get_data()
            reward_batch[i] = self.history[i].get_data()
            terminal1_batch[i] = self.history[i].get_data()

        result = {
            'obss0': array_min2d(obs0_batch),
            'obss1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        self.currentMemEps.append(obs0, action, reward, obs1, terminal1, training)

    def appendEps(self):
        self.history.append(copy.copy(self.currentMemEps))
        self.currentMemEps.reset()

    @property
    def nb_entries(self):
        return len(self.history)

class MemoryEps(object):
    def __init__(self, horizonlen, action_shape, observation_shape):
        self.horizonlen = horizonlen
        self.start = 0
        self.length = 0

        self.observations0 = np.zeros((horizonlen,) + observation_shape).astype('float32')
        self.actions = np.zeros((horizonlen,) + action_shape).astype('float32')
        self.rewards = np.zeros((horizonlen, 1,)).astype('float32')
        self.terminals1 = np.zeros((horizonlen, 1,)).astype('float32')
        self.observations1 = np.zeros((horizonlen,) + observation_shape).astype('float32')

    def reset(self):
        self.observations0.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.terminals1.fill(0)
        self.observations1.fill(0)

    def get_data(self):
        result = {
            'obss0': array_min2d(self.observations0),
            'obss1': array_min2d(self.observations1),
            'rewards': array_min2d(self.rewards),
            'actions': array_min2d(self.actions),
            'terminals1': array_min2d(self.terminals1),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        if self.length < self.horizonlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.horizonlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.horizonlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.observations0[(self.start + self.length - 1) % self.horizonlen] = obs0
        self.actions[(self.start + self.length - 1) % self.horizonlen] = action
        self.rewards[(self.start + self.length - 1) % self.horizonlen] = reward
        self.observations1[(self.start + self.length - 1) % self.horizonlen] = obs1
        self.terminals1[(self.start + self.length - 1) % self.horizonlen] = terminal1

