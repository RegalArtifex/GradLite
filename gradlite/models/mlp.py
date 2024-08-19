from gradlite import Layer, Value
from typing import List


class MLP:
    def __init__(self, n_inputs: int, layers: List[int], activations: List[str]) -> None:
        layers_with_input = [n_inputs] + layers
        if activations:
            self.layers = [Layer(layers_with_input[i], layers_with_input[i+1], activations[i]) for i in range(len(layers))]
        else:
            self.layers = [Layer(layers_with_input[i], layers_with_input[i+1]) for i in range(len(layers))]

    def __call__(self, x) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def __str__(self) -> str:
        _str = ""
        for layer in self.layers:
            _str += f"{layer}\n"
        return _str

    def __repr__(self) -> str:
        _str = ""
        for layer in self.layers:
            _str += f"{layer}\n"
        return _str
    
    def summary(self):
        print("---------------------------------------")
        print("Model Summary:")
        print("---------------------------------------")
        for i, layer in enumerate(self.layers):
            print(f"{i}: {layer}\n")

    @property
    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters]
