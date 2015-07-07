"""
TabooSampling: Draw a sample from given N distinct items
  however previous M samples are not chosen.

N must be greater than  2.
M is ceil((N - 2) / 2) by default.
When N = 3, it remembers previous 1 sample and avoid it.
When N = 5, it remembers previous 2 samples and avoid them.
"""

from math import ceil
from random import choice

class TabooSampling(object):
    def __init__(self, samples):
        self.samples = set(samples)
        self.N = len(samples)
        assert self.N > 2
        self.M = int(ceil((self.N - 2) / 2.0))
        self.taboos = [None] * self.M
        self.taboo_index = 0

    def choice(self):
        sample = choice(list(self.samples - set(self.taboos)))
        self.taboos[self.taboo_index] = sample
        self.taboo_index = (self.taboo_index + 1) % self.M
        return sample
