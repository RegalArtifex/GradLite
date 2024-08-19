import random
from typing import List
from gradlite.core.value import Value


class Neuron:
    def __init__(self, n_inputs, activation='tanh') -> None:
        self.w = [Value(random.uniform(-1, 1), activation) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1), activation)
    
    def __call__(self, x):
        out = sum( (wi * xi for wi, xi in zip(self.w, x)), self.b)
        activated_out = out.activate()
        return activated_out
    
    @property
    def parameters(self) -> List[Value]:
        return self.w + [self.b]
