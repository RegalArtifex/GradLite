from __future__ import annotations
from typing import List, Union
import math


class Value:
    def __init__(self, data: float, activation='tanh', prev: set=(), op: str='', grad: float=0.0, label: str='') -> None:
        self.data = data
        self.prev = set(prev)
        self.op = op
        self.grad = grad
        self.label = label
        self._backward = lambda: None
        self.activation = activation

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    ##############################################################
    # Basic Operations
    ##############################################################

    def __add__(self, other: Value) -> Value:
        other = Value(other) if isinstance(other, float) or isinstance(other, int) else other
        new_data = self.data + other.data
        op = '+'
        out = Value(data=new_data, activation=self.activation, prev=(self, other), op=op)

        # Defining the closure to compute the gradient
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        # Assigning it to out's backward
        out._backward = _backward
        return out

    def __radd__(self, other: Union[int, float]) -> Value:
        return self.__add__(other)

    def __mul__(self, other: Union[Value, int, float]) -> Value:
        other = Value(other) if isinstance(other, float) or isinstance(other, int) else other
        new_data = self.data * other.data
        op = '*'
        out = Value(data=new_data, activation=self.activation, prev=(self, other), op=op)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Union[int, float]) -> Value:
        return self.__mul__(other)
    
    def __truediv__(self, other) -> Value:
        return self * other**-1
    
    def __rtruediv__(self, other) -> Value:
        return self**-1 * other
    
    def __neg__(self) -> Value:
        return -1 * self
    
    def __sub__(self, other) -> Value:
        return self + (-other)
    
    def __rsub__(self, other) -> Value:
        return other + (-self)

    def __pow__(self, other: Union[int, float]) -> Value:
        assert isinstance(other, (float, int)), "Only Int and Float powers are supported!"

        out = Value(self.data**other, self.activation, (self, ), f"**{other}")

        def _backward():
            self.grad += out.data * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    ##############################################################
    # Special Math Functions (Activations)
    ##############################################################

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, prev=(self,), op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return  out
    
    def relu(self) -> Value:
        out = Value(max(0, self.data), prev=(self,), op='relu')

        def _backward():
            self.grad += (self.grad > 0) * out.grad
        
        out._backward = _backward
        return out

    def exp(self) -> Value:
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def activate(self) -> Value:
        if self.activation == 'tanh':
            return self.tanh()
        elif self.activation == 'relu':
            return self.relu()

    ##############################################################
    # Gradient Computation
    ##############################################################

    def backward(self):
        topology: List[Value] = []
        visited = set()

        # Function to visit each child and add it to topology
        # (Topological Traversal of Directed Acyclic Graphs)
        def build_topology(node: Value):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    build_topology(child)
                topology.append(node)

        # Build Topology
        build_topology(self)

        # Initialize the gradient
        self.grad = 1.0
        for node in reversed(topology):
            node._backward()
