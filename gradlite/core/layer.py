from gradlite.core.neuron import Neuron
from gradlite.core.value import Value
from typing import List


class Layer:
    def __init__(self, n_inputs, n_outpus, activation='tanh') -> None:
        self.n_inputs = n_inputs
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outpus)]
        self.activation = activation
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def __str__(self):
        return f"Layer({len(self.neurons)} neurons, n_inputs={self.n_inputs}) -> {self.activation}()"
    
    def __repr__(self):
        return f"Layer({len(self.neurons)} neurons, n_inputs={self.n_inputs}) -> {self.activation}()"
    
    @property
    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters]
