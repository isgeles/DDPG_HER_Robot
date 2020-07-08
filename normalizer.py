import threading
import numpy as np
import torch


class Normalizer(object):
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        """Online standard normalization.
        """
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        self.mean = np.zeros(self.size, dtype=np.float32)
        self.std = np.ones(self.size, dtype=np.float32)
        self.sum = np.zeros(self.size, dtype=np.float32)
        self.sum_squared = np.zeros(self.size, dtype=np.float32)
        self.count = 1

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sum_squared = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        self.lock = threading.Lock()

    def normalize(self, v):
        return np.clip((v - self.mean) / self.std, -self.clip_range, self.clip_range).astype(np.float32)

    def update(self, v):
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sum_squared += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        with self.lock:
            self.count += self.local_count
            self.sum += self.local_sum
            self.sum_squared += self.local_sum_squared

            self.local_count[:] = 0
            self.local_sum[:] = 0
            self.local_sum_squared[:] = 0

        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(np.square(self.eps),
                                      self.sum_squared / self.count - np.square(self.sum/self.count)))

    def load_normalizer(self, path):
        [self.mean, self.std, self.sum, self.sum_squared, self.count] = torch.load(path)

    def save_normalizer(self, path):
        torch.save([self.mean, self.std, self.sum, self.sum_squared, self.count], path)

