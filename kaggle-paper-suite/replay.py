"""Original batch-pair replay semantics with a step-keyed random stream."""
import numpy as np


class ReplayPool:
    def __init__(self, size, batch_shape):
        self.values = np.zeros((size, 2) + tuple(batch_shape), dtype=np.float32)
        self.count = 0

    def query(self, a, b, seed):
        size = len(self.values)
        if not size:
            return a, b
        if self.count < size:
            self.values[self.count, 0] = a
            self.values[self.count, 1] = b
            self.count += 1
            return a, b
        rng = np.random.RandomState(seed)
        if rng.rand() > .5:
            ia, ib = int(rng.rand() * size), int(rng.rand() * size)
            old_a, old_b = self.values[ia, 0].copy(), self.values[ib, 1].copy()
            self.values[ia, 0], self.values[ib, 1] = a, b
            return old_a, old_b
        return a, b
