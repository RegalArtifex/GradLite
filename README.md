![image](./assets/banner.png)

# 🪶 GradLite

GradLite is a teeny tiny practice implementation of **Autograd** from scratch, built without relying on any third-party autograd libraries. This project is implemented using only basic Python datatypes and the Python Standard Library.

The primary goal of this project is to deepen my understanding of **backpropagation** and **neural network fundamentals** by building an autograd system from the ground up.

## ✨ Features

- **Lightweight**: No dependencies on external autograd libraries.
- **Educational**: Designed to demonstrate the inner workings of backpropagation and autograd.
- **Pythonic**: Utilizes pure Python, keeping the code easy to follow and understand.
- **Object Oriented**: Designed using Object Oriented Programming concepts so that the code is modular.

## 📚 Overview

GradLite provides a minimalistic implementation of autograd, a core component in most modern machine learning frameworks. The code showcases how gradients are computed and propagated through a network during the training process, emphasizing the fundamental concepts of:

- **Gradient Calculation**: Computing the Gradient using the **chain rule**.
- **Forward Pass**: Computing the output of a neural network.
- **Backward Pass**: Propagating gradients back through the network to update parameters.

## 🚀 Getting Started

To explore and experiment with this implementation, simply clone the repository and run the provided examples.

```bash
git clone https://github.com/yourusername/GradLite.git
cd GradLite
python examples.py
```

## 🛠️ Usage

The implementation is straightforward and can be easily extended or modified. The core functionality revolves around a few simple classes and methods that handle the forward and backward passes of the network.

## 🧠 Understanding the Code

The code is heavily commented to aid understanding. Key components include:

- **Value Class**: Represents the core data structure (which is a value with autograd) with support for forward and backward operations.
- **Neuron Class**: Represents the fundamental unit of a Neural Network. This data structure depends on the `Value` class to represent the **weights** and **bias**.
- **Layer Class**: Represents a **Layer** in a Neural Network. Layer can be understood as a group of **neurons** belonging to a single layer in a Neural Network.
- **Neural Network Example**: A simple neural network ([MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)) implementation to demonstrate how GradLite can be used in practice.

## 🤝 Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Contributions are welcome!

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## 🧑‍💻 Author

GradLite was created by  [Raghul P](https://github.com/RegalArtifex). Connect with me on [LinkedIn](https://www.linkedin.com/in/raghul-p) for more insights into AI and deep learning!

### 🙏 Acknowledgments

This project was inspired by Andrej Karpathy's tutorial, ["The spelled-out intro to neural networks and backpropagation: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0) on YouTube. His work provided valuable insights into building an autograd system from scratch.
