import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """
        Creates a replay.
        @param buffer_shapes: (dict of ints) the shape for all buffers that are used in the replay buffer
        @param size_in_transitions: (int) the size of the buffer, measured in transitions
        @param T: (int) the time horizon for episodes (episode length)
        @param sample_transitions: a function that samples from the replay buffer
        """
        self.size = size_in_transitions // T
        self.sample_transitions = sample_transitions
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        self.current_size = 0
        self.lock = threading.Lock()

    def sample(self, batch_size):
        """
        Returns a dict from the saved samples in the buffer: {key: array(batch_size x shapes[key])}
        @param batch_size: size of batch for learning step of agent
        @return: sample transitions
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        return self.sample_transitions(buffers, batch_size)

    def store_episode(self, episode_batch):
        """
        Store episode from batch to buffer. Observation 'o' is of size T+1, others are of size T.
        @param episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_size = len(episode_batch['u'])
        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

    def _get_storage_idx(self, inc=1):
        """
        Returns to you the indexes where you will write in the buffer.
        These are consecutive until you hit the end, then they are random.
        @param inc: incrementing stepsize in buffer, usually 1
        @return: index where to store in buffer
        """
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        self.current_size = min(self.size, self.current_size+inc)
        return idx
