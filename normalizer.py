import threading
import numpy as np
import torch


class Normalizer(object):
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        """
        Online standard normalization.
        @param size: size of array to normalize
        @param eps: minimal eps for std of distribution
        @param clip_range: range to clip values of normalized array, so they are in [-clip_range, clip_range]
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
        """
        Normalizing array v with current stats for mean/std
        @param v: array to normalize
        @return: normalized and clipped array
        """
        return np.clip((v - self.mean) / self.std, -self.clip_range, self.clip_range).astype(np.float32)

    def update(self, v):
        """
        update internal stats when another array is added
        @param v: array added to for online updating mean/std
        """
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sum_squared += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        """
        Recomputing the stats used for normalization.
        """
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
        """
        Load values of mean, std, sum, sum_squared, self.count so that Normalizer object can be used for evaluation.
        @param path: where to load values from
        """
        [self.mean, self.std, self.sum, self.sum_squared, self.count] = torch.load(path)

    def save_normalizer(self, path):
        """
        Save values of mean, std, sum, sum_squared, self.count so that Normalizer object can be reconstructed later.
        @param path: where to store values
        """
        torch.save([self.mean, self.std, self.sum, self.sum_squared, self.count], path)

