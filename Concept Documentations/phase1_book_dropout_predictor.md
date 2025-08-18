# Personalized AI — Recommendations & Emotion Recognition

## Implementation

Binary classification,
Multilayer Perceptron
PyTorch from scratch,
ReLU/Sigmoid layers,
Loss,
Evaluation,
Accuracy.

## Documentation

Neuralnetworks:
Bias- added
Weights- multiplied

Backpropagation: calculating values of b and w

Tensors:store and manipulate data with GPU acceleration and automatic differentiation

Scalars=0D Tensors,
Arrays=1D Tensors.
Images=2D Tensors,
Video=nD Tensors.

MLP: built from layers of perceptrons between input and output

- linear transformations and nonlinear activations.
- takes four input features, passes them through a hidden layer with 4 neurons and a ReLU activation, then outputs a single raw value (called a logit)
- Linear layer performs a weighted sum of inputs plus a bias
- ReLU(x)=max(0,x)
- All negative values get squashed to 0
- Positive values stay unchanged

- The sigmoid function is applied to the logits during evaluation to obtain probabilities, then convert them to predictions using a threshold (typically 0.5).
- gradients are cleared (optimizer.zero_grad())
- loss.backward(): computes the gradient of the loss w.r.t each parameter in your model for next
- optimizer.step(): update model weights

Epochs: Repeat the process multiple times until accuracy improves.
