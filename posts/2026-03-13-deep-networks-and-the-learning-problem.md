---
title: "Deep Networks and the Learning Problem"
date: "2026-03-13"
excerpt: "How stacking neurons into layers creates universal function approximators, and the mathematical machinery - backpropagation and gradient descent - that makes them learn."
author: "Chase Dovey"
tags: ["AI", "Deep Learning"]
draft: true
---

## Introduction

The perceptron can only learn linearly separable functions. XOR defeats it. Any problem requiring a curved or complex decision boundary is out of reach.

The solution was obvious in principle: stack multiple layers of neurons so that earlier layers transform the inputs into a representation where the final layer can draw a linear boundary. A hidden layer can learn to re-map the input space, bending and folding it until the classes become separable. This is the multilayer perceptron (MLP).

The problem was training. With a single layer, the perceptron learning rule tells you exactly how to adjust each weight: the error at the output directly touches every weight. With multiple layers, the output error does not directly touch the weights in earlier layers. How do you assign credit (or blame) for the output error to weights buried deep in the network?

The answer is backpropagation: apply the chain rule of calculus to propagate error gradients backward through the network, layer by layer. Combined with gradient descent to update the weights, this gives us the complete learning machinery for deep networks.

## The Multilayer Perceptron

An MLP has three types of layers:

- **Input layer**: Receives the raw features. No computation happens here.
- **Hidden layers**: One or more layers of neurons that transform the input. Each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function.
- **Output layer**: Produces the final prediction.

Every neuron in one layer connects to every neuron in the next layer. This is called a **fully connected** or **dense** architecture.

<svg viewBox="0 0 460 220" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="220" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">multilayer perceptron (MLP)</text>
  <!-- input layer -->
  <circle cx="60" cy="55" r="14" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="60" y="59" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x1</text>
  <circle cx="60" cy="110" r="14" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="60" y="114" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x2</text>
  <circle cx="60" cy="165" r="14" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="60" y="169" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x3</text>
  <!-- hidden layer 1 -->
  <circle cx="180" cy="45" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="180" y="49" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h1</text>
  <circle cx="180" cy="95" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="180" y="99" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h2</text>
  <circle cx="180" cy="145" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="180" y="149" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h3</text>
  <circle cx="180" cy="195" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="180" y="199" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h4</text>
  <!-- hidden layer 2 -->
  <circle cx="300" cy="70" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="300" y="74" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h5</text>
  <circle cx="300" cy="120" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="300" y="124" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h6</text>
  <circle cx="300" cy="170" r="14" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="300" y="174" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h7</text>
  <!-- output layer -->
  <circle cx="400" cy="110" r="14" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="400" y="114" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">y</text>
  <!-- input to hidden1 edges (all connections) -->
  <line x1="74" y1="55" x2="166" y2="45" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="55" x2="166" y2="95" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="55" x2="166" y2="145" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="55" x2="166" y2="195" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="110" x2="166" y2="45" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="110" x2="166" y2="95" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="110" x2="166" y2="145" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="110" x2="166" y2="195" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="165" x2="166" y2="45" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="165" x2="166" y2="95" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="165" x2="166" y2="145" stroke="#555" stroke-width="0.5"/>
  <line x1="74" y1="165" x2="166" y2="195" stroke="#555" stroke-width="0.5"/>
  <!-- hidden1 to hidden2 edges -->
  <line x1="194" y1="45" x2="286" y2="70" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="45" x2="286" y2="120" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="45" x2="286" y2="170" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="95" x2="286" y2="70" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="95" x2="286" y2="120" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="95" x2="286" y2="170" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="145" x2="286" y2="70" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="145" x2="286" y2="120" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="145" x2="286" y2="170" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="195" x2="286" y2="70" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="195" x2="286" y2="120" stroke="#555" stroke-width="0.5"/>
  <line x1="194" y1="195" x2="286" y2="170" stroke="#555" stroke-width="0.5"/>
  <!-- hidden2 to output edges -->
  <line x1="314" y1="70" x2="386" y2="110" stroke="#555" stroke-width="0.5"/>
  <line x1="314" y1="120" x2="386" y2="110" stroke="#555" stroke-width="0.5"/>
  <line x1="314" y1="170" x2="386" y2="110" stroke="#555" stroke-width="0.5"/>
  <!-- layer labels -->
  <text x="60" y="205" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">input</text>
  <text x="180" y="215" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">hidden 1</text>
  <text x="300" y="205" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">hidden 2</text>
  <text x="400" y="145" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">output</text>
  <!-- signal animation -->
  <circle r="3" fill="#4ade80" opacity="0">
    <animateMotion path="M74,110 L166,95" dur="4s" repeatCount="indefinite" keyTimes="0;0.2;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.18;0.2;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M194,95 L286,120" dur="4s" repeatCount="indefinite" keyTimes="0;0.3;0.5;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.29;0.3;0.48;0.5;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#fbbf24" opacity="0">
    <animateMotion path="M314,120 L386,110" dur="4s" repeatCount="indefinite" keyTimes="0;0.6;0.8;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.59;0.6;0.78;0.8;1" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>

### The Forward Pass as Matrix Multiplication

For a single hidden layer, the forward pass is:

```
h = activation(W1 . x + b1)     # hidden layer
y = activation(W2 . h + b2)     # output layer
```

Where:
- `W1` is the weight matrix from input to hidden (shape: hidden_size x input_size)
- `b1` is the bias vector for the hidden layer
- `W2` is the weight matrix from hidden to output
- `b2` is the bias vector for the output layer

This generalizes to any number of layers:

```
a0 = x                               # input
z1 = W1 . a0 + b1                    # pre-activation, layer 1
a1 = activation(z1)                   # post-activation, layer 1
z2 = W2 . a1 + b2                    # pre-activation, layer 2
a2 = activation(z2)                   # post-activation, layer 2
...
y = aL                                # output = final activation
```

Each layer applies a linear transformation (matrix multiply + bias) followed by a nonlinear activation function. The linear part rotates, scales, and shifts the data. The nonlinear part bends and folds it. Together, they can approximate any continuous function.

### The Universal Approximation Theorem

In 1989, George Cybenko proved that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of R^n to arbitrary accuracy, given enough neurons.

This theorem says MLPs are **universal function approximators**. It does not say they are easy to train, or that the required number of neurons is reasonable, or that gradient descent will find the right weights. It is an existence proof: the representational capacity is there.

In practice, deep networks (many layers) tend to learn more efficiently than wide ones (many neurons in one layer). Depth allows hierarchical feature extraction: early layers learn simple features, later layers compose them into complex ones.

## Activation Functions: Why Nonlinearity Matters

Without an activation function, an MLP is just a series of matrix multiplications. And the composition of linear functions is linear: `W2 . (W1 . x + b1) + b2 = W2.W1.x + W2.b1 + b2 = W'.x + b'`. No matter how many layers you stack, the network can only compute linear functions. It collapses to a single-layer perceptron.

The activation function introduces nonlinearity, which is what gives deep networks their power. Every activation function used in practice is a simple nonlinear function applied element-wise to the pre-activation values.

### Sigmoid

```
sigmoid(z) = 1 / (1 + exp(-z))
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

Range: (0, 1). Squashes any input to a probability-like value. Was the standard for decades. Problem: gradients vanish for large |z| because the derivative approaches 0. This kills learning in deep networks.

### Tanh

```
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
tanh'(z) = 1 - tanh(z)^2
```

Range: (-1, 1). Zero-centered, which helps optimization. But still saturates for large |z|, causing vanishing gradients.

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, else 0
```

Range: [0, infinity). Does not saturate for positive values, so gradients flow freely. Computationally trivial. The default choice for most architectures since ~2012. Downside: "dying ReLU" problem where neurons with consistently negative inputs never activate and stop learning entirely.

### GELU (Gaussian Error Linear Unit)

```
GELU(z) = z * Phi(z)
```

Where `Phi` is the cumulative distribution function of the standard normal distribution. GELU smoothly gates values rather than hard-thresholding at zero. Used in BERT, GPT, and most transformer architectures.

### SiLU / Swish

```
SiLU(z) = z * sigmoid(z)
```

Smooth, non-monotonic, self-gated. Used in many modern architectures including LLaMA.

<svg viewBox="0 0 460 200" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="200" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">activation functions</text>
  <!-- axes -->
  <line x1="60" y1="100" x2="430" y2="100" stroke="#555" stroke-width="0.5"/>
  <line x1="200" y1="30" x2="200" y2="180" stroke="#555" stroke-width="0.5"/>
  <text x="435" y="104" fill="#999" font-family="monospace" font-size="7">z</text>
  <text x="204" y="28" fill="#999" font-family="monospace" font-size="7">f(z)</text>
  <!-- tick marks -->
  <text x="200" y="112" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">0</text>
  <!-- Sigmoid (red-ish) -->
  <path d="M60,170 Q100,168 130,160 Q160,140 180,120 Q195,105 200,100 Q205,95 220,80 Q240,60 260,45 Q290,35 340,32 L430,30" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4"/>
  <!-- ReLU (green) -->
  <path d="M60,100 L200,100 L430,30" fill="none" stroke="#4ade80" stroke-width="2"/>
  <!-- GELU (cyan) -->
  <path d="M60,105 Q100,106 130,107 Q160,108 180,104 Q190,100 200,96 Q220,75 260,50 Q300,32 340,28 L430,20" fill="none" stroke="#22d3ee" stroke-width="1.5"/>
  <!-- legend -->
  <line x1="300" y1="145" x2="320" y2="145" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="325" y="149" fill="#ef4444" font-family="monospace" font-size="8">sigmoid</text>
  <line x1="300" y1="160" x2="320" y2="160" stroke="#4ade80" stroke-width="2"/>
  <text x="325" y="164" fill="#4ade80" font-family="monospace" font-size="8">ReLU</text>
  <line x1="300" y1="175" x2="320" y2="175" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="325" y="179" fill="#22d3ee" font-family="monospace" font-size="8">GELU</text>
</svg>

## Loss Functions: Measuring Error

Training a neural network requires a way to measure how wrong its predictions are. This measurement is the **loss function** (also called the cost function or objective function). The goal of training is to minimize the loss.

### Mean Squared Error (MSE)

```
L = (1/n) * sum((yi - ti)^2)
```

Where `yi` is the prediction and `ti` is the target. Penalizes large errors quadratically. Used for regression tasks.

### Cross-Entropy Loss

```
L = -(1/n) * sum(ti * log(yi) + (1-ti) * log(1-yi))
```

For binary classification. When the prediction is confident and correct, the loss is near zero. When the prediction is confident and wrong, the loss approaches infinity. This harsh penalty for confident mistakes drives faster learning than MSE for classification.

For multi-class classification with softmax output:

```
L = -(1/n) * sum_i(sum_c(t_ic * log(y_ic)))
```

Where `c` indexes over classes.

## The Chain Rule: Foundation of Backpropagation

Before we can derive backpropagation, we need the chain rule of calculus. If `y = f(g(x))`, then:

```
dy/dx = (dy/dg) * (dg/dx) = f'(g(x)) * g'(x)
```

For a composition of many functions `y = f1(f2(f3(...fn(x)...)))`:

```
dy/dx = f1' * f2' * f3' * ... * fn'
```

The chain rule lets us compute how the output of a composed function changes when we change the input, by multiplying the local derivatives at each step. This is exactly what we need for deep networks: the network is a composition of layer functions, and we need to know how the loss changes when we change each weight.

## Backpropagation

Backpropagation, published by Rumelhart, Hinton, and Williams in 1986, is the chain rule applied systematically to neural networks. It computes the gradient of the loss with respect to every weight in the network in a single backward pass.

### The Computational Graph

Think of the network as a directed graph where each node is an operation (multiply, add, activate) and edges carry values. The forward pass computes output from input. The backward pass computes gradients from output to input.

<svg viewBox="0 0 480 180" xmlns="http://www.w3.org/2000/svg" style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="480" height="180" rx="8" fill="#181818"/>
  <text x="240" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">forward and backward pass</text>
  <!-- forward pass (top) -->
  <text x="45" y="42" fill="#4ade80" font-family="monospace" font-size="8">input</text>
  <rect x="28" y="48" width="36" height="22" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="46" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">x</text>
  <line x1="64" y1="59" x2="95" y2="59" stroke="#4ade80" stroke-width="1"/>
  <text x="80" y="53" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">fwd</text>
  <rect x="95" y="48" width="50" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="120" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">W1.x+b1</text>
  <line x1="145" y1="59" x2="168" y2="59" stroke="#4ade80" stroke-width="1"/>
  <rect x="168" y="48" width="40" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="188" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">relu</text>
  <line x1="208" y1="59" x2="230" y2="59" stroke="#4ade80" stroke-width="1"/>
  <rect x="230" y="48" width="50" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="255" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">W2.h+b2</text>
  <line x1="280" y1="59" x2="303" y2="59" stroke="#4ade80" stroke-width="1"/>
  <rect x="303" y="48" width="40" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="323" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">sig</text>
  <line x1="343" y1="59" x2="368" y2="59" stroke="#4ade80" stroke-width="1"/>
  <rect x="368" y="48" width="36" height="22" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="386" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">y</text>
  <line x1="404" y1="59" x2="425" y2="59" stroke="#4ade80" stroke-width="1"/>
  <rect x="425" y="48" width="36" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="443" y="63" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">L</text>
  <!-- backward pass (bottom) -->
  <rect x="425" y="110" width="36" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="443" y="125" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">dL</text>
  <line x1="425" y1="121" x2="404" y2="121" stroke="#ef4444" stroke-width="1"/>
  <text x="415" y="140" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">bwd</text>
  <rect x="368" y="110" width="36" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="386" y="125" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">dL/dy</text>
  <line x1="368" y1="121" x2="343" y2="121" stroke="#ef4444" stroke-width="1"/>
  <rect x="303" y="110" width="40" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="323" y="125" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">dL/dz2</text>
  <line x1="303" y1="121" x2="280" y2="121" stroke="#ef4444" stroke-width="1"/>
  <rect x="230" y="110" width="50" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="255" y="125" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">dL/dW2</text>
  <line x1="230" y1="121" x2="208" y2="121" stroke="#ef4444" stroke-width="1"/>
  <rect x="168" y="110" width="40" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="188" y="125" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">dL/dh</text>
  <line x1="168" y1="121" x2="145" y2="121" stroke="#ef4444" stroke-width="1"/>
  <rect x="95" y="110" width="50" height="22" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="120" y="125" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">dL/dW1</text>
  <!-- animated forward pulse -->
  <circle r="3" fill="#4ade80" opacity="0">
    <animateMotion path="M46,59 L120,59 L188,59 L255,59 L323,59 L386,59 L443,59" dur="5s" repeatCount="indefinite" keyTimes="0;0.05;0.35;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.33;0.35;1" dur="5s" repeatCount="indefinite"/>
  </circle>
  <!-- animated backward pulse -->
  <circle r="3" fill="#ef4444" opacity="0">
    <animateMotion path="M443,121 L386,121 L323,121 L255,121 L188,121 L120,121" dur="5s" repeatCount="indefinite" keyTimes="0;0.45;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.44;0.45;0.83;0.85;1" dur="5s" repeatCount="indefinite"/>
  </circle>
  <!-- labels -->
  <text x="240" y="170" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">forward: compute outputs left to right</text>
  <text x="240" y="155" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">backward: compute gradients right to left</text>
</svg>

### Deriving Backpropagation

Consider a two-layer network with MSE loss. We want `dL/dW1` and `dL/dW2`.

**Forward pass:**
```
z1 = W1 . x + b1
a1 = relu(z1)
z2 = W2 . a1 + b2
y = sigmoid(z2)
L = (1/2)(y - t)^2
```

**Backward pass (chain rule applied right to left):**

Step 1: Loss gradient with respect to output:
```
dL/dy = y - t
```

Step 2: Output gradient with respect to pre-activation z2:
```
dL/dz2 = dL/dy * dy/dz2 = (y - t) * sigmoid'(z2) = (y - t) * y * (1 - y)
```

Step 3: Gradient with respect to W2 (this is what we use to update W2):
```
dL/dW2 = dL/dz2 * dz2/dW2 = dL/dz2 * a1
```

Step 4: Gradient with respect to hidden activation a1 (needed to continue backward):
```
dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * W2
```

Step 5: Gradient with respect to pre-activation z1:
```
dL/dz1 = dL/da1 * da1/dz1 = dL/da1 * relu'(z1)
```

Where `relu'(z1) = 1` if `z1 > 0`, else `0`.

Step 6: Gradient with respect to W1:
```
dL/dW1 = dL/dz1 * dz1/dW1 = dL/dz1 * x
```

Each step multiplies the incoming gradient by the local derivative. This is backpropagation: the chain rule applied layer by layer, from output to input.

### Why Activation Derivatives Matter

Look at step 5. The gradient flowing backward through a ReLU is either the full gradient (if `z1 > 0`) or zero (if `z1 <= 0`). ReLU either passes the gradient through unchanged or kills it. This is why ReLU trains faster than sigmoid: sigmoid's derivative is always less than 0.25, so it attenuates the gradient at every layer. Stack 10 layers of sigmoid and the gradient shrinks by a factor of `0.25^10 = 0.000001`. This is the **vanishing gradient problem**.

## Gradient Descent

Backpropagation computes the gradient. Gradient descent uses it to update the weights:

```
W = W - alpha * dL/dW
```

The negative sign means we move in the direction that decreases the loss. The learning rate `alpha` controls the step size.

### Vanilla Gradient Descent

Compute the gradient using the **entire training set**, then take one step. This gives the exact gradient but is slow for large datasets. Each update requires processing every example.

### Stochastic Gradient Descent (SGD)

Compute the gradient using a **single random example**, then take one step. The gradient estimate is noisy but unbiased. Many small, noisy steps converge faster than few large, exact steps for large datasets. In practice, **mini-batch SGD** uses a small batch (32-512 examples) as a compromise between noise and efficiency.

### Momentum

SGD can oscillate in narrow valleys of the loss landscape. Momentum adds a velocity term that accumulates past gradients:

```
v = beta * v - alpha * dL/dW
W = W + v
```

The velocity `v` builds up in directions of consistent gradient and dampens oscillations. Like a ball rolling downhill that builds speed and rolls through small bumps.

### Adam (Adaptive Moment Estimation)

Adam combines momentum with per-parameter adaptive learning rates:

```
m = beta1 * m + (1 - beta1) * dL/dW       # first moment (mean of gradient)
v = beta2 * v + (1 - beta2) * (dL/dW)^2    # second moment (variance of gradient)
m_hat = m / (1 - beta1^t)                   # bias correction
v_hat = v / (1 - beta2^t)                   # bias correction
W = W - alpha * m_hat / (sqrt(v_hat) + eps)
```

Parameters with consistent gradients get effectively larger steps. Parameters with noisy gradients get smaller steps. Adam is the default optimizer for most modern architectures.

<svg viewBox="0 0 460 200" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="200" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">gradient descent on a loss landscape</text>
  <!-- loss surface contours -->
  <ellipse cx="230" cy="120" rx="180" ry="60" fill="none" stroke="#333" stroke-width="1"/>
  <ellipse cx="230" cy="120" rx="140" ry="45" fill="none" stroke="#444" stroke-width="1"/>
  <ellipse cx="230" cy="120" rx="100" ry="30" fill="none" stroke="#555" stroke-width="1"/>
  <ellipse cx="230" cy="120" rx="60" ry="18" fill="none" stroke="#666" stroke-width="1"/>
  <ellipse cx="230" cy="120" rx="20" ry="6" fill="none" stroke="#888" stroke-width="1"/>
  <!-- minimum marker -->
  <circle cx="230" cy="120" r="3" fill="#fbbf24"/>
  <text x="230" y="140" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">minimum</text>
  <!-- SGD path (zigzag) -->
  <path d="M80,55 L130,80 L100,100 L150,105 L120,115 L170,118 L190,120 L210,119 L225,120" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <text x="75" y="50" fill="#ef4444" font-family="monospace" font-size="7">SGD</text>
  <!-- Adam path (smoother) -->
  <path d="M380,170 L350,155 L320,140 L290,132 L265,126 L245,122 L235,121" fill="none" stroke="#4ade80" stroke-width="1.5"/>
  <text x="390" y="178" fill="#4ade80" font-family="monospace" font-size="7">Adam</text>
  <!-- animated SGD dot -->
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M80,55 L130,80 L100,100 L150,105 L120,115 L170,118 L190,120 L210,119 L225,120" dur="5s" repeatCount="indefinite" keyTimes="0;0.05;0.9;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.88;0.9;1" dur="5s" repeatCount="indefinite"/>
  </circle>
  <!-- animated Adam dot -->
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M380,170 L350,155 L320,140 L290,132 L265,126 L245,122 L235,121" dur="5s" repeatCount="indefinite" keyTimes="0;0.05;0.9;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.88;0.9;1" dur="5s" repeatCount="indefinite"/>
  </circle>
</svg>

## Vanishing and Exploding Gradients

Backpropagation multiplies gradients through each layer. If these multiplied values are consistently less than 1, the gradient shrinks exponentially as it propagates backward. After many layers, the gradient reaching the first layer is effectively zero. The early layers stop learning. This is the **vanishing gradient problem**.

Conversely, if the multiplied values are consistently greater than 1, the gradient grows exponentially. Weights receive enormous updates and diverge to infinity. This is the **exploding gradient problem**.

Solutions developed over the years:

- **ReLU activation**: Gradients are either 0 or 1, avoiding the attenuation caused by sigmoid/tanh derivatives.
- **Careful weight initialization**: Xavier initialization (for sigmoid/tanh) and He initialization (for ReLU) set initial weights to maintain gradient variance across layers.
- **Batch normalization**: Normalizes the inputs to each layer, stabilizing the gradient flow.
- **Residual connections**: Add the layer's input to its output (`y = f(x) + x`), creating a gradient highway that bypasses the vanishing problem. This is what makes very deep networks (100+ layers) trainable.
- **Gradient clipping**: Cap the gradient magnitude to prevent explosions.

## Putting It All Together

The complete training loop for an MLP:

```
Initialize weights using He initialization
Choose optimizer (typically Adam)
Set learning rate (typically 1e-3 to 1e-4)

For each epoch:
    Shuffle training data
    For each mini-batch:
        1. Forward pass: compute predictions
        2. Compute loss
        3. Backward pass: compute gradients via backpropagation
        4. Update weights using optimizer
    Evaluate on validation set
    Adjust learning rate if needed
```

This loop, forward-loss-backward-update, is the same loop used by every neural network trained today. Transformers, CNNs, diffusion models, they all use the same machinery. The architecture changes. The loss function changes. The optimizer settings change. But the training loop is the same one that Rumelhart, Hinton, and Williams described in 1986.

## Key Takeaways

The MLP stacks neurons into layers, enabling nonlinear decision boundaries and universal function approximation. Backpropagation applies the chain rule to compute gradients efficiently through the network. Gradient descent (especially Adam) uses these gradients to iteratively improve the weights. The vanishing gradient problem limited early deep networks until ReLU, careful initialization, and residual connections solved it. The forward-loss-backward-update loop is the universal training algorithm for neural networks.
