import threading
import numpy as np
import torch


class Normalizer(object):
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        """A normalizer that ensures that observations are approximately distributed according to
        a standard Normal distribution (i.e. have mean zero and variance one).

        Args:
            size (int): the size of the observation to be normalized
            eps (float): a small constant that avoids underflows
            default_clip_range (float): normalized observations are clipped to be in
                [-default_clip_range, default_clip_range]
        """
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range  # always 5

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        self.lock = threading.Lock()

        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_count = 1

    def update(self, v):
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def normalize(self, v):
        clip_range = self.default_clip_range
        return np.clip((v - self.running_mean) / self.running_std, -clip_range, clip_range).astype(np.float32)

    def recompute_stats(self):
        with self.lock:
            self.running_count += self.local_count
            self.running_sum += self.local_sum
            self.running_sum_sq += self.local_sumsq

            # reset local
            self.local_count[:] = 0
            self.local_sum[:] = 0
            self.local_sumsq[:] = 0

        self.running_mean = self.running_sum / self.running_count
        self.running_std = np.sqrt(np.maximum(np.square(self.eps),
                                              self.running_sum_sq / self.running_count
                                              - np.square(self.running_sum/self.running_count)))

    def load_normalizer(self, path):
        [self.running_mean, self.running_std, self.running_sum, self.running_sum_sq, self.running_count] = \
            torch.load(path)

    def save_normalizer(self, path):
        torch.save([self.running_mean, self.running_std, self.running_sum, self.running_sum_sq, self.running_count]
                   , path)

