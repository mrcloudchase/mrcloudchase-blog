---
title: "Deep Networks and the Learning Problem"
date: "2026-03-06"
excerpt: "How stacking neurons creates universal function approximators, and the mathematical machinery that finally solved the credit assignment problem that froze AI for fifteen years."
author: "Chase Dovey"
tags: ["AI", "Deep Learning"]
draft: true
---

## Introduction

The previous post ended with a problem expressed in code. I built `mlp()`, a two-layer network that solves XOR when every weight is set by hand. Then I tried to train it with `train()`, the perceptron learning rule, and it failed. The learning rule computes `error = target - output`, but hidden neurons have no targets. The output neuron knows it got XOR wrong. The hidden neurons do not know what they should have produced. That is the credit assignment problem.

Rosenblatt spent the last decade of his life trying to solve it. He built multi-layer networks with random hidden projections and trainable output layers, but he never found a way to propagate learning into the hidden connections. Minsky and Papert proved the single-layer model was fundamentally limited. The field froze for over fifteen years.

The solution was hiding in calculus the entire time. Three changes turn a single-layer perceptron into a trainable deep network:

1. Replace the step function with a smooth, differentiable activation
2. Define a continuous measure of how wrong the output is (a loss function)
3. Use the chain rule to propagate that error backward through every layer

Each change is simple on its own. Together, they solve the problem that killed neural network research and launch the field that eventually produces modern deep learning. Let me build each piece.

## The Step Function Problem

The perceptron uses a step function: output 1 if `z >= 0`, output 0 otherwise. The output is binary. There is nothing in between.

For a single layer, this works fine. The perceptron learning rule does not need to know *how wrong* the output is, only *that* it is wrong. The update `w = w + eta * (t - y) * x` uses the raw error (which is -1, 0, or +1) as a correction signal. That is enough when every weight connects directly to the output.

For hidden layers, it is not enough. I need to answer a different question: "How much should I change this hidden weight to reduce the error at the output?" That question requires a derivative. If I change a hidden weight by a small amount, how much does the output change? For the step function, the answer is almost always zero. The output is flat (derivative zero) everywhere except at `z = 0`, where it jumps discontinuously (derivative undefined). A tiny change in any hidden weight produces either no change in the output or an infinite change. There is no gradient to follow.

```
Step function:
    y = 1 if z >= 0, else 0

    Derivative:
    dy/dz = 0    everywhere except z = 0
    dy/dz = undefined    at z = 0
```

The fix: replace the step function with something smooth. Something whose output changes gradually as the input changes, so that the derivative exists everywhere and tells me which direction to push.

## Activation Functions

### Sigmoid: The Historical Fix

The function that made deep learning possible is the logistic sigmoid:

```
sigmoid(z) = 1 / (1 + exp(-z))
```

It looks like the step function with the hard edge smoothed out. For large positive `z`, the output approaches 1. For large negative `z`, it approaches 0. At `z = 0`, the output is exactly 0.5. The transition between 0 and 1 is smooth, and the derivative exists everywhere:

```
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

The derivative has a convenient property: it can be computed from the output alone. If `a = sigmoid(z)`, then `sigmoid'(z) = a * (1 - a)`. No need to store `z`. This matters for efficiency during backpropagation.

The maximum derivative is 0.25, at `z = 0`. For large `|z|`, the derivative approaches zero. The function *saturates*: when the neuron is confidently on or confidently off, the gradient vanishes. This will become a serious problem in deep networks, but for now it is enough that the derivative exists at all.

This is the activation function that [Rumelhart, Hinton, and Williams used in 1986](https://www.nature.com/articles/323533a0) to demonstrate backpropagation on multi-layer networks. It is the function that ended the AI winter.

### ReLU: The Modern Default

Sigmoid's saturation problem motivated the search for alternatives. The breakthrough came from a deceptively simple function:

```
relu(z) = max(0, z)
```

Zero for negative inputs. Identity for positive inputs. The derivative is 0 or 1. No saturation for positive values, so gradients flow without shrinking. [Nair and Hinton demonstrated in 2010](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) that ReLU significantly improved training of deep networks.

The tradeoff: neurons with negative pre-activations have zero gradient. If a neuron's weighted sum goes negative and stays there, it stops learning entirely. These "dead neurons" are the cost of ReLU's simplicity.

### GELU: The Transformer's Choice

```
gelu(z) = z * Phi(z)
```

Where `Phi(z)` is the standard Gaussian cumulative distribution function. GELU smoothly gates the input by its own probability of being positive under a Gaussian distribution. For large positive values, GELU approaches the identity (like ReLU). For large negative values, it approaches zero (like ReLU). But the transition is smooth, with a small negative dip near `z = -0.75`. This is the activation used in GPT, BERT, and most modern transformers.

<svg viewBox="0 0 480 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Comparison of four activation functions: step (binary jump), sigmoid (smooth S-curve), ReLU (bent line at zero), and GELU (smooth bent line with slight negative dip). Each shown on the same axes to highlight the progression from non-differentiable to smooth." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Activation function comparison: step, sigmoid, ReLU, GELU</title>
  <rect width="480" height="220" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">activation functions</text>
  <!-- axes -->
  <line x1="60" y1="190" x2="230" y2="190" stroke="#555" stroke-width="1"/>
  <line x1="145" y1="30" x2="145" y2="190" stroke="#555" stroke-width="1"/>
  <text x="145" y="206" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">z</text>
  <!-- y=0 and y=1 reference lines -->
  <line x1="62" y1="110" x2="228" y2="110" stroke="#333" stroke-width="0.5" stroke-dasharray="2"/>
  <text x="56" y="114" fill="#555" font-family="monospace" font-size="7" text-anchor="end">0</text>
  <line x1="62" y1="50" x2="228" y2="50" stroke="#333" stroke-width="0.5" stroke-dasharray="2"/>
  <text x="56" y="54" fill="#555" font-family="monospace" font-size="7" text-anchor="end">1</text>
  <!-- Step function (gray, dashed) -->
  <polyline points="62,110 145,110 145,50 228,50" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="4"/>
  <!-- Sigmoid (cyan) -->
  <polyline points="62,108 72,107 82,106 92,104 102,101 112,96 122,88 132,77 137,70 142,63 147,57 152,52 162,47 172,44 182,42 192,41 202,40 212,40 222,40 228,40" fill="none" stroke="#22d3ee" stroke-width="1.5"/>
  <!-- ReLU (green) -->
  <polyline points="62,110 82,110 102,110 122,110 142,110 145,110 155,98 165,86 175,74 185,62 195,50 205,38 215,26" fill="none" stroke="#4ade80" stroke-width="1.5"/>
  <!-- GELU (yellow) -->
  <polyline points="62,110 72,110 82,110 92,111 102,112 112,113 122,113 127,112 132,111 137,110 142,109 147,107 152,103 157,98 162,92 167,84 172,76 177,68 182,60 192,46 202,36 212,28 222,22" fill="none" stroke="#fbbf24" stroke-width="1.5"/>
  <!-- Legend -->
  <text x="290" y="50" fill="#999" font-family="monospace" font-size="9">Activation Functions:</text>
  <line x1="290" y1="66" x2="310" y2="66" stroke="#888" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="316" y="70" fill="#888" font-family="monospace" font-size="8">step (perceptron)</text>
  <line x1="290" y1="84" x2="310" y2="84" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="316" y="88" fill="#22d3ee" font-family="monospace" font-size="8">sigmoid (1986)</text>
  <line x1="290" y1="102" x2="310" y2="102" stroke="#4ade80" stroke-width="1.5"/>
  <text x="316" y="106" fill="#4ade80" font-family="monospace" font-size="8">ReLU (2010)</text>
  <line x1="290" y1="120" x2="310" y2="120" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="316" y="124" fill="#fbbf24" font-family="monospace" font-size="8">GELU (transformers)</text>
  <!-- Annotation -->
  <text x="350" y="160" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">the step function has</text>
  <text x="350" y="173" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">zero gradient everywhere</text>
  <text x="350" y="186" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">smooth functions have</text>
  <text x="350" y="199" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">gradients calculus can use</text>
</svg>

*Figure 1: Four activation functions on the same axes. The step function (dashed gray) is the perceptron's original activation, binary with zero derivative. Sigmoid (cyan) was the first differentiable replacement, enabling backpropagation in 1986. ReLU (green) solved the vanishing gradient problem with a constant derivative of 1 for positive inputs. GELU (yellow) is the smooth variant used in modern transformers.*

## The MLP: Architecture and Forward Pass

With a differentiable activation function, I can revisit the multilayer perceptron from the previous post. The architecture is the same: input layer, one or more hidden layers, output layer. Each layer computes a weighted sum plus bias, then applies an activation function. The difference is that the activation is now sigmoid instead of step, and the output is a continuous value between 0 and 1 instead of a hard 0 or 1.

For a two-layer network (one hidden layer, one output layer), the forward pass is:

```
Hidden layer:   z1 = W1 @ x + b1      a1 = sigmoid(z1)
Output layer:   z2 = W2 @ a1 + b2     y  = sigmoid(z2)
```

The complete function is a composition: `f(x) = sigmoid(W2 @ sigmoid(W1 @ x + b1) + b2)`. Each layer transforms its input. The composition of transformations can represent nonlinear decision boundaries, exactly the kind the perceptron could not express.

How powerful is this? In 1989, George Cybenko proved the [universal approximation theorem](https://link.springer.com/article/10.1007/BF02551274): a single hidden layer with enough neurons and a non-constant, bounded, continuous activation function (like sigmoid) can approximate any continuous function on a compact set to arbitrary accuracy. Kurt Hornik extended this in 1991 to show the result holds for a broader class of activations. The MLP is not just more powerful than a perceptron. It is a universal function approximator.

The catch is "enough neurons." A single hidden layer can represent anything in theory, but in practice it may need an astronomically large number of neurons. Deep networks, those with many layers, can often represent the same functions far more efficiently. Depth is not about what you can represent, it is about how efficiently you can represent it.

<svg viewBox="0 0 480 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Multilayer perceptron with 2 input nodes, 2 hidden nodes with sigmoid activation, and 1 output node. Matrix dimensions annotated on connections. Animated signal pulse flows left to right through the network." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>MLP architecture: 2 inputs, 2 hidden neurons (sigmoid), 1 output neuron (sigmoid), with matrix dimensions and animated forward pass</title>
  <rect width="480" height="230" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">multilayer perceptron (2-2-1)</text>
  <!-- input layer -->
  <text x="80" y="42" text-anchor="middle" fill="#4ade80" font-family="monospace" font-size="9">input</text>
  <circle cx="80" cy="80" r="18" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="80" y="84" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x1</text>
  <circle cx="80" cy="155" r="18" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="80" y="159" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x2</text>
  <!-- hidden layer -->
  <text x="240" y="42" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9">hidden</text>
  <circle cx="240" cy="80" r="18" fill="#333" stroke="#22d3ee" stroke-width="1.5">
    <animate attributeName="stroke" values="#22d3ee;#22d3ee;#fff;#22d3ee;#22d3ee" keyTimes="0;0.30;0.35;0.45;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <text x="240" y="76" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">sigmoid</text>
  <text x="240" y="87" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h1</text>
  <circle cx="240" cy="155" r="18" fill="#333" stroke="#22d3ee" stroke-width="1.5">
    <animate attributeName="stroke" values="#22d3ee;#22d3ee;#fff;#22d3ee;#22d3ee" keyTimes="0;0.30;0.35;0.45;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <text x="240" y="151" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">sigmoid</text>
  <text x="240" y="162" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h2</text>
  <!-- output layer -->
  <text x="400" y="42" text-anchor="middle" fill="#fbbf24" font-family="monospace" font-size="9">output</text>
  <circle cx="400" cy="118" r="18" fill="#333" stroke="#fbbf24" stroke-width="1.5">
    <animate attributeName="stroke" values="#fbbf24;#fbbf24;#fbbf24;#fff;#fbbf24;#fbbf24" keyTimes="0;0.50;0.55;0.60;0.70;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <text x="400" y="114" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">sigmoid</text>
  <text x="400" y="125" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">y</text>
  <!-- connections: input to hidden -->
  <line x1="98" y1="80" x2="222" y2="80" stroke="#555" stroke-width="1"/>
  <line x1="98" y1="80" x2="222" y2="155" stroke="#555" stroke-width="1"/>
  <line x1="98" y1="155" x2="222" y2="80" stroke="#555" stroke-width="1"/>
  <line x1="98" y1="155" x2="222" y2="155" stroke="#555" stroke-width="1"/>
  <!-- connections: hidden to output -->
  <line x1="258" y1="80" x2="382" y2="118" stroke="#555" stroke-width="1"/>
  <line x1="258" y1="155" x2="382" y2="118" stroke="#555" stroke-width="1"/>
  <!-- matrix dimension labels -->
  <text x="160" y="70" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">W1: 2x2</text>
  <text x="320" y="92" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">W2: 1x2</text>
  <!-- forward pass pulse: input to hidden -->
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M98,80 L222,80" dur="4s" repeatCount="indefinite" keyTimes="0;0.15;0.30;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.14;0.15;0.28;0.30;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M98,155 L222,155" dur="4s" repeatCount="indefinite" keyTimes="0;0.15;0.30;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.14;0.15;0.28;0.30;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <!-- forward pass pulse: hidden to output -->
  <circle r="4" fill="#22d3ee" opacity="0">
    <animateMotion path="M258,80 L382,118" dur="4s" repeatCount="indefinite" keyTimes="0;0.45;0.60;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.44;0.45;0.58;0.60;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#22d3ee" opacity="0">
    <animateMotion path="M258,155 L382,118" dur="4s" repeatCount="indefinite" keyTimes="0;0.45;0.60;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.44;0.45;0.58;0.60;1" dur="4s" repeatCount="indefinite"/>
  </circle>
  <!-- bias labels -->
  <text x="240" y="210" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">+ b1 (2x1)</text>
  <text x="400" y="160" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">+ b2 (1x1)</text>
</svg>

*Figure 2: A 2-2-1 MLP for XOR. Two inputs feed into two hidden neurons (each applying sigmoid), which feed into one output neuron (also sigmoid). The weight matrices W1 (2x2) and W2 (1x2) are the learnable parameters. The forward pass flows left to right: inputs are transformed by the hidden layer into a new representation, then the output layer classifies that representation.*

## The Learning Problem: Loss Functions

Now I have a differentiable architecture that can represent nonlinear functions. The next question: how do I measure "wrong" precisely enough for calculus to work with?

The perceptron used `error = target - output`, which takes values in {-1, 0, 1}. With sigmoid outputs between 0 and 1, I need a continuous measure of distance between the output and the target.

### Mean Squared Error

The simplest option: square the difference.

```
L = (target - output)^2
```

Derivative with respect to the output: `dL/dy = -2(target - output)`. Simple and intuitive. But for classification with sigmoid, MSE has a problem. When the output is confidently wrong (sigmoid near 0 but target is 1, or near 1 but target is 0), sigmoid's derivative is tiny, so the combined gradient is small. The model learns slowly exactly when it is most wrong.

### Cross-Entropy: The Right Loss for Classification

```
L = -[t * log(y) + (1-t) * log(1-y)]
```

Where `t` is the target (0 or 1) and `y` is the sigmoid output. When `t = 1`, the loss is `-log(y)`: the closer `y` is to 1, the smaller the loss. When `y` approaches 0 (confidently wrong), the loss goes to infinity. The loss heavily penalizes confident wrong answers.

The derivative of cross-entropy with respect to the sigmoid output is `(y - t) / (y * (1 - y))`. Combined with sigmoid's derivative `y * (1 - y)`, the compound gradient for the output layer simplifies to:

```
dL/dz = y - t
```

The `y * (1 - y)` terms cancel. There is no saturation. The gradient is simply the difference between the output and the target. This is why cross-entropy became the default loss for classification: the gradient signal stays strong regardless of how wrong the output is.

## The Chain Rule: Why Calculus Solves Credit Assignment

This is the conceptual core of the post. Everything before this was setup. The credit assignment problem, the one that stumped Rosenblatt for a decade and froze the field, reduces to a single idea from calculus.

The question is: given a loss `L` computed at the output, how much is each weight in each layer responsible? Formally, I need `dL/dw` for every weight `w` in every layer.

Consider a two-layer network. The loss depends on the output `y`, which depends on the output pre-activation `z2`, which depends on the hidden activations `a1`, which depends on the hidden pre-activations `z1`, which depends on the first-layer weights `W1`. It is a chain of dependencies:

```
W1 -> z1 -> a1 -> z2 -> y -> L
```

The chain rule says: if `L` depends on `y` which depends on `z2` which depends on `a1` which depends on `z1` which depends on `W1`, then:

```
dL/dW1 = dL/dy * dy/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
```

Each factor in that product is a local derivative. `dL/dy` is the loss derivative. `dy/dz2` is the activation derivative. `dz2/da1` is just the output weight matrix. `da1/dz1` is the hidden activation derivative. `dz1/dW1` is just the input. Every factor can be computed from values already available during the forward pass.

The credit assignment problem was never about missing information. The forward pass computes everything needed. The chain rule tells you how to use it. The line that solves the problem is:

```
dL/da1 = W2^T @ dL/dz2
```

The output weight matrix `W2`, transposed, distributes the output error back to each hidden neuron in proportion to that neuron's connection strength. A hidden neuron connected to the output by a large positive weight gets a large share of the blame. A neuron connected by a small weight gets little blame. A neuron connected by a negative weight gets blame in the opposite direction.

This is exactly what Rosenblatt could not do. His learning rule needed a target for each neuron. The chain rule replaces targets with gradients. Hidden neurons do not need targets because the chain rule computes their contribution to the error directly from the network's own weights.

<svg viewBox="0 0 480 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Computational graph of a small neural network showing the forward pass computing values left to right and the backward pass propagating gradients right to left via the chain rule. Each node shows the local derivative used in the chain." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Computational graph: forward pass computes values left to right, backward pass propagates gradients right to left using the chain rule</title>
  <rect width="480" height="250" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">computational graph: forward + backward</text>
  <!-- forward pass (top row) -->
  <text x="240" y="38" text-anchor="middle" fill="#4ade80" font-family="monospace" font-size="8">forward pass (compute values)</text>
  <!-- nodes -->
  <rect x="20" y="52" width="52" height="30" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="46" y="71" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">x</text>
  <rect x="106" y="52" width="52" height="30" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="132" y="71" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">z1</text>
  <rect x="192" y="52" width="52" height="30" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="218" y="71" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">a1</text>
  <rect x="278" y="52" width="52" height="30" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="304" y="71" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">z2</text>
  <rect x="364" y="52" width="52" height="30" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="390" y="71" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">y</text>
  <rect x="440" y="52" width="30" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="455" y="71" fill="#ef4444" font-family="monospace" font-size="9" text-anchor="middle">L</text>
  <!-- forward arrows -->
  <line x1="72" y1="67" x2="106" y2="67" stroke="#4ade80" stroke-width="1"/>
  <polygon points="104,64 110,67 104,70" fill="#4ade80"/>
  <line x1="158" y1="67" x2="192" y2="67" stroke="#22d3ee" stroke-width="1"/>
  <polygon points="190,64 196,67 190,70" fill="#22d3ee"/>
  <line x1="244" y1="67" x2="278" y2="67" stroke="#22d3ee" stroke-width="1"/>
  <polygon points="276,64 282,67 276,70" fill="#22d3ee"/>
  <line x1="330" y1="67" x2="364" y2="67" stroke="#fbbf24" stroke-width="1"/>
  <polygon points="362,64 368,67 362,70" fill="#fbbf24"/>
  <line x1="416" y1="67" x2="440" y2="67" stroke="#ef4444" stroke-width="1"/>
  <polygon points="438,64 444,67 438,70" fill="#ef4444"/>
  <!-- operation labels -->
  <text x="89" y="58" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">W1x+b1</text>
  <text x="175" y="58" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">sigmoid</text>
  <text x="261" y="58" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">W2a+b2</text>
  <text x="347" y="58" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">sigmoid</text>
  <text x="430" y="58" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">loss</text>
  <!-- backward pass (bottom row) -->
  <text x="240" y="115" text-anchor="middle" fill="#ef4444" font-family="monospace" font-size="8">backward pass (propagate gradients)</text>
  <!-- gradient nodes -->
  <rect x="440" y="128" width="30" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="455" y="147" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="364" y="128" width="52" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="390" y="147" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">dL/dy</text>
  <rect x="278" y="128" width="52" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="304" y="147" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">dL/dz2</text>
  <rect x="192" y="128" width="52" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="218" y="147" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">dL/da1</text>
  <rect x="106" y="128" width="52" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="132" y="147" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">dL/dz1</text>
  <rect x="20" y="128" width="52" height="30" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="46" y="147" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">dL/dW1</text>
  <!-- backward arrows -->
  <line x1="440" y1="143" x2="416" y2="143" stroke="#ef4444" stroke-width="1"/>
  <polygon points="418,140 412,143 418,146" fill="#ef4444"/>
  <line x1="364" y1="143" x2="330" y2="143" stroke="#ef4444" stroke-width="1"/>
  <polygon points="332,140 326,143 332,146" fill="#ef4444"/>
  <line x1="278" y1="143" x2="244" y2="143" stroke="#ef4444" stroke-width="1"/>
  <polygon points="246,140 240,143 246,146" fill="#ef4444"/>
  <line x1="192" y1="143" x2="158" y2="143" stroke="#ef4444" stroke-width="1"/>
  <polygon points="160,140 154,143 160,146" fill="#ef4444"/>
  <line x1="106" y1="143" x2="72" y2="143" stroke="#ef4444" stroke-width="1"/>
  <polygon points="74,140 68,143 74,146" fill="#ef4444"/>
  <!-- local derivative labels on backward arrows -->
  <text x="430" y="136" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">dL/dy</text>
  <text x="347" y="136" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">y(1-y)</text>
  <text x="261" y="136" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">W2^T</text>
  <text x="175" y="136" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">a1(1-a1)</text>
  <text x="89" y="136" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">x^T</text>
  <!-- chain rule formula -->
  <text x="240" y="185" text-anchor="middle" fill="#ccc" font-family="monospace" font-size="8">dL/dW1 = dL/dy * dy/dz2 * dz2/da1 * da1/dz1 * dz1/dW1</text>
  <text x="240" y="200" text-anchor="middle" fill="#999" font-family="monospace" font-size="8">each factor is a local derivative</text>
  <!-- animated gradient pulse -->
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M455,143 L416,143 L330,143 L244,143 L158,143 L72,143" dur="5s" repeatCount="indefinite" keyTimes="0;0.1;0.3;0.5;0.7;0.9;1" keyPoints="0;0;0.2;0.4;0.6;0.8;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0.9;0.9;0.9;0;0" keyTimes="0;0.09;0.1;0.3;0.5;0.7;0.88;0.9;1" dur="5s" repeatCount="indefinite"/>
  </circle>
  <!-- KEY INSIGHT annotation -->
  <rect x="135" y="212" width="210" height="28" rx="4" fill="none" stroke="#22d3ee" stroke-width="1" stroke-dasharray="3"/>
  <text x="240" y="230" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="8">W2^T distributes blame to hidden neurons</text>
</svg>

*Figure 3: The computational graph of a two-layer network. Top row: the forward pass computes values from left to right. Bottom row: the backward pass propagates gradients from right to left by multiplying local derivatives at each step. The chain rule connects the output error to every weight in the network. The key step is W2^T, which distributes blame from the output to hidden neurons in proportion to their connection strength.*

## Backpropagation: The Algorithm

### History

[Paul Werbos described backpropagation in his 1974 PhD thesis *Beyond Regression*](https://gwern.net/doc/ai/nn/1974-werbos.pdf). The work went largely unnoticed. In 1986, [David Rumelhart, Geoffrey Hinton, and Ronald Williams published *Learning Representations by Back-Propagating Errors*](https://www.nature.com/articles/323533a0) in Nature, demonstrating backpropagation on multi-layer networks and showing that hidden layers learn useful intermediate representations. This paper ended the AI winter and revived neural network research.

The algorithm has two phases: a forward pass that computes the output, and a backward pass that computes the gradient of the loss with respect to every weight.

### Step by Step

```
1. FORWARD PASS
   For each layer (input to output):
       z = W @ a_prev + b          (weighted sum)
       a = sigmoid(z)               (activation)
       Store z and a for use in backward pass

2. COMPUTE LOSS
   L = -[t * log(y) + (1-t) * log(1-y)]    (cross-entropy)

3. BACKWARD PASS
   Output layer:
       delta = y - t                         (cross-entropy + sigmoid)
       dL/dW = delta @ a_prev^T              (weight gradient)
       dL/db = delta                          (bias gradient)

   For each hidden layer (output to input):
       delta = (W_next^T @ delta_next) * sigmoid'(a)    (propagate + activate)
       dL/dW = delta @ a_prev^T                          (weight gradient)
       dL/db = delta                                      (bias gradient)

4. UPDATE
   For every weight and bias:
       w = w - eta * dL/dw
       b = b - eta * dL/db
```

The key line is `delta = (W_next^T @ delta_next) * sigmoid'(a)`. This propagates the error from the next layer back to the current layer. The transposed weight matrix distributes blame. The activation derivative gates it. This is the credit assignment step that Rosenblatt could never perform.

### Worked Example

I will trace one forward and backward pass through a 2-2-1 network with specific weights. This is the XOR architecture from the perceptron post, but now with sigmoid activations and backpropagation.

**Setup:**

```
W1 = [[0.5, -0.3],     b1 = [0.1, -0.2]
      [0.2,  0.8]]

W2 = [[0.6, -0.4]]     b2 = [0.3]

Input: x = [1, 0]      Target: t = 1
```

**Forward pass:**

```
Hidden layer:
    z1[0] = 0.5*1 + (-0.3)*0 + 0.1 = 0.6
    z1[1] = 0.2*1 + 0.8*0 + (-0.2) = 0.0
    a1[0] = sigmoid(0.6) = 0.6457
    a1[1] = sigmoid(0.0) = 0.5000

Output layer:
    z2 = 0.6*0.6457 + (-0.4)*0.5000 + 0.3 = 0.4874
    y  = sigmoid(0.4874) = 0.6195

Loss:
    L = -[1*log(0.6195) + 0*log(0.3805)] = 0.4789
```

The network outputs 0.6195. The target is 1. The loss is 0.4789. Now I propagate the error backward.

**Backward pass:**

```
Output delta (cross-entropy + sigmoid simplification):
    delta2 = y - t = 0.6195 - 1 = -0.3805

Output weight gradients:
    dL/dW2[0] = delta2 * a1[0] = -0.3805 * 0.6457 = -0.2457
    dL/dW2[1] = delta2 * a1[1] = -0.3805 * 0.5000 = -0.1903
    dL/db2    = -0.3805
```

Now the key step. Propagate the error to the hidden layer:

```
Error distributed to hidden neurons:
    dL/da1[0] = W2[0] * delta2 = 0.6 * (-0.3805) = -0.2283
    dL/da1[1] = W2[1] * delta2 = (-0.4) * (-0.3805) = 0.1522

Hidden deltas (multiply by sigmoid derivative):
    delta1[0] = -0.2283 * a1[0]*(1 - a1[0]) = -0.2283 * 0.2288 = -0.0522
    delta1[1] =  0.1522 * a1[1]*(1 - a1[1]) =  0.1522 * 0.2500 =  0.0381

Hidden weight gradients:
    dL/dW1[0][0] = delta1[0] * x[0] = -0.0522 * 1 = -0.0522
    dL/dW1[0][1] = delta1[0] * x[1] = -0.0522 * 0 =  0.0000
    dL/dW1[1][0] = delta1[1] * x[0] =  0.0381 * 1 =  0.0381
    dL/dW1[1][1] = delta1[1] * x[1] =  0.0381 * 0 =  0.0000
```

Every weight in the network now has a gradient. The output weights get gradients directly from the output error. The hidden weights get gradients via the chain rule, with `W2^T` distributing blame and `sigmoid'` gating it. The weight update `w = w - eta * dL/dw` nudges every weight in the direction that reduces the loss. No hand-design. No per-neuron targets. Just the chain rule.

<svg viewBox="0 0 480 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Animated two-phase diagram of a neural network. Phase 1: green signal pulses flow forward from inputs through hidden to output, computing values. Phase 2: red gradient pulses flow backward from the loss through output and hidden layers, computing weight updates." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Forward pass (green, left to right) followed by backward pass (red, right to left) through a 2-2-1 network</title>
  <rect width="480" height="280" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">backpropagation: forward then backward</text>
  <!-- network structure -->
  <!-- input nodes -->
  <circle cx="60" cy="90" r="16" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="60" y="94" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">x1</text>
  <circle cx="60" cy="170" r="16" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="60" y="174" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">x2</text>
  <!-- hidden nodes -->
  <circle cx="200" cy="90" r="16" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="200" y="94" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h1</text>
  <circle cx="200" cy="170" r="16" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="200" y="174" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">h2</text>
  <!-- output node -->
  <circle cx="340" cy="130" r="16" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="340" y="134" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">y</text>
  <!-- loss node -->
  <rect x="400" y="118" width="36" height="24" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="418" y="134" fill="#ef4444" font-family="monospace" font-size="9" text-anchor="middle">L</text>
  <!-- connections -->
  <line x1="76" y1="90" x2="184" y2="90" stroke="#555" stroke-width="1"/>
  <line x1="76" y1="90" x2="184" y2="170" stroke="#555" stroke-width="1"/>
  <line x1="76" y1="170" x2="184" y2="90" stroke="#555" stroke-width="1"/>
  <line x1="76" y1="170" x2="184" y2="170" stroke="#555" stroke-width="1"/>
  <line x1="216" y1="90" x2="324" y2="130" stroke="#555" stroke-width="1"/>
  <line x1="216" y1="170" x2="324" y2="130" stroke="#555" stroke-width="1"/>
  <line x1="356" y1="130" x2="400" y2="130" stroke="#555" stroke-width="1"/>
  <!-- PHASE 1: Forward pass pulses (green, 0-40% of cycle) -->
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M76,90 L184,90" dur="8s" repeatCount="indefinite" keyTimes="0;0.05;0.20;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.04;0.05;0.18;0.20;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M76,170 L184,170" dur="8s" repeatCount="indefinite" keyTimes="0;0.05;0.20;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.04;0.05;0.18;0.20;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#22d3ee" opacity="0">
    <animateMotion path="M216,90 L324,130" dur="8s" repeatCount="indefinite" keyTimes="0;0.22;0.37;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.21;0.22;0.35;0.37;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#22d3ee" opacity="0">
    <animateMotion path="M216,170 L324,130" dur="8s" repeatCount="indefinite" keyTimes="0;0.22;0.37;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.21;0.22;0.35;0.37;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#fbbf24" opacity="0">
    <animateMotion path="M356,130 L400,130" dur="8s" repeatCount="indefinite" keyTimes="0;0.38;0.44;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.37;0.38;0.42;0.44;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <!-- PHASE 2: Backward pass pulses (red, 55-95% of cycle) -->
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M400,130 L356,130" dur="8s" repeatCount="indefinite" keyTimes="0;0.55;0.61;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.54;0.55;0.59;0.61;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M324,130 L216,90" dur="8s" repeatCount="indefinite" keyTimes="0;0.63;0.78;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.62;0.63;0.76;0.78;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M324,130 L216,170" dur="8s" repeatCount="indefinite" keyTimes="0;0.63;0.78;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.62;0.63;0.76;0.78;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M184,90 L76,90" dur="8s" repeatCount="indefinite" keyTimes="0;0.80;0.93;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.79;0.80;0.91;0.93;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M184,170 L76,170" dur="8s" repeatCount="indefinite" keyTimes="0;0.80;0.93;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.79;0.80;0.91;0.93;1" dur="8s" repeatCount="indefinite"/>
  </circle>
  <!-- phase labels -->
  <text x="130" y="46" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.03;0.05;0.44;0.46;1" dur="8s" repeatCount="indefinite"/>
    forward: compute values -->
  </text>
  <text x="300" y="46" fill="#ef4444" font-family="monospace" font-size="9" text-anchor="middle">
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.53;0.55;0.93;0.95;1" dur="8s" repeatCount="indefinite"/>
    backward: compute gradients
  </text>
  <!-- gradient annotations that appear during backward phase -->
  <text x="360" y="112" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.54;0.55;0.93;0.95;1" dur="8s" repeatCount="indefinite"/>
    y - t
  </text>
  <text x="270" y="100" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.62;0.63;0.93;0.95;1" dur="8s" repeatCount="indefinite"/>
    W2^T @ delta
  </text>
  <text x="130" y="75" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.79;0.80;0.93;0.95;1" dur="8s" repeatCount="indefinite"/>
    delta * x^T
  </text>
  <!-- update text -->
  <text x="240" y="260" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">every weight gets a gradient. every weight learns.</text>
</svg>

*Figure 4: Backpropagation in action. Phase 1 (green/cyan): the forward pass computes values from inputs through hidden to output. Phase 2 (red): the backward pass propagates gradients from the loss back through the network. The output error y - t is distributed to hidden neurons via W2^T, then to input weights via the chain rule. Every weight in every layer gets a gradient and updates accordingly.*

## Gradient Descent

Backpropagation computes the gradient `dL/dw` for every weight. The gradient points in the direction of steepest increase of the loss. To reduce the loss, I move in the opposite direction:

```
w = w - eta * dL/dw
```

This is gradient descent. The learning rate `eta` controls step size. But there are several variants, and the choice matters.

**Vanilla gradient descent** computes the gradient over the entire training set, then takes one step. Stable but slow. For large datasets, computing the full gradient before each step is prohibitive.

**Stochastic gradient descent (SGD)** computes the gradient on a single random training example, then takes a step. Noisy but fast. Each step is cheap, and the noise actually helps escape shallow local minima. This is what makes training on millions of examples feasible.

**Mini-batch SGD** is the practical compromise: compute the gradient on a small batch of examples (typically 32 to 512), then step. It reduces noise while keeping updates frequent. This is the standard in practice.

**Momentum** adds inertia to the updates:

```
v = beta * v + dL/dw
w = w - eta * v
```

The velocity `v` accumulates a running average of past gradients. When gradients point in the same direction across multiple steps, the velocity builds up and the steps get larger. When gradients oscillate, the velocity dampens them. Like a ball rolling downhill: it accelerates on consistent slopes and resists zig-zagging.

**Adam** ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)) adapts the learning rate for each weight individually. It maintains running estimates of the first moment (mean) and second moment (uncentered variance) of the gradient, with bias correction for the early steps. Weights with consistently large gradients get smaller learning rates. Weights with small or noisy gradients get larger ones. Adam is the default optimizer for most modern deep learning.

<svg viewBox="0 0 480 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Loss landscape shown as 2D contour plot with four optimizer paths: vanilla gradient descent (slow straight), SGD (noisy zigzag), momentum (smooth curve), and Adam (efficient diagonal path). Each optimizer animated as a moving dot following its characteristic trajectory to the minimum." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Loss landscape with optimizer paths: vanilla GD, SGD, momentum, and Adam converging toward the minimum</title>
  <rect width="480" height="230" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">loss landscape + optimizer paths</text>
  <!-- contour lines (concentric ellipses around minimum) -->
  <ellipse cx="310" cy="150" rx="130" ry="70" fill="none" stroke="#333" stroke-width="0.8"/>
  <ellipse cx="310" cy="150" rx="100" ry="55" fill="none" stroke="#333" stroke-width="0.8"/>
  <ellipse cx="310" cy="150" rx="70" ry="40" fill="none" stroke="#444" stroke-width="0.8"/>
  <ellipse cx="310" cy="150" rx="45" ry="25" fill="none" stroke="#555" stroke-width="0.8"/>
  <ellipse cx="310" cy="150" rx="20" ry="12" fill="none" stroke="#666" stroke-width="0.8"/>
  <!-- minimum marker -->
  <circle cx="310" cy="150" r="3" fill="#fbbf24"/>
  <text x="310" y="170" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">minimum</text>
  <!-- Vanilla GD path (slow, straight-ish) -->
  <polyline points="190,60 210,70 230,82 250,95 265,108 278,120 290,132 300,140 306,146 310,150" fill="none" stroke="#888" stroke-width="1.5"/>
  <circle r="4" fill="#888">
    <animateMotion path="M190,60 L210,70 L230,82 L250,95 L265,108 L278,120 L290,132 L300,140 L306,146 L310,150" dur="6s" repeatCount="indefinite" calcMode="linear"/>
  </circle>
  <!-- SGD path (noisy zigzag) -->
  <polyline points="190,80 215,65 225,95 250,75 260,110 280,90 295,125 305,138 308,148 310,150" fill="none" stroke="#ef4444" stroke-width="1" stroke-dasharray="3"/>
  <circle r="4" fill="#ef4444">
    <animateMotion path="M190,80 L215,65 L225,95 L250,75 L260,110 L280,90 L295,125 L305,138 L308,148 L310,150" dur="6s" repeatCount="indefinite" calcMode="linear"/>
  </circle>
  <!-- Momentum path (smooth curve, slight overshoot) -->
  <polyline points="195,45 230,60 265,90 290,120 305,145 315,155 312,152 310,150" fill="none" stroke="#22d3ee" stroke-width="1.5"/>
  <circle r="4" fill="#22d3ee">
    <animateMotion path="M195,45 L230,60 L265,90 L290,120 L305,145 L315,155 L312,152 L310,150" dur="6s" repeatCount="indefinite" calcMode="linear"/>
  </circle>
  <!-- Adam path (efficient diagonal) -->
  <polyline points="200,55 240,80 275,110 295,135 305,145 310,150" fill="none" stroke="#4ade80" stroke-width="1.5"/>
  <circle r="4" fill="#4ade80">
    <animateMotion path="M200,55 L240,80 L275,110 L295,135 L305,145 L310,150" dur="6s" repeatCount="indefinite" calcMode="linear"/>
  </circle>
  <!-- Legend -->
  <text x="50" y="140" fill="#888" font-family="monospace" font-size="8">vanilla GD</text>
  <text x="50" y="155" fill="#ef4444" font-family="monospace" font-size="8">SGD (noisy)</text>
  <text x="50" y="170" fill="#22d3ee" font-family="monospace" font-size="8">momentum</text>
  <text x="50" y="185" fill="#4ade80" font-family="monospace" font-size="8">Adam</text>
</svg>

*Figure 5: A loss landscape with four optimizer paths. Vanilla gradient descent (gray) takes slow, steady steps. SGD (red, dashed) is noisy, zig-zagging toward the minimum. Momentum (cyan) follows a smooth curve with slight overshoot, then corrects. Adam (green) takes the most efficient path by adapting its learning rate per weight.*

## The Vanishing Gradient Problem

Backpropagation solved the credit assignment problem. But it introduced a new one.

The chain rule multiplies local derivatives along the path from output to input. With sigmoid activations, each local derivative is `sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))`, which has a maximum value of 0.25 at `z = 0`. In a 10-layer network, the gradient at the first layer passes through 10 sigmoid derivatives:

```
0.25 * 0.25 * 0.25 * ... (10 times) = 0.25^10 = 0.00000095
```

The gradient vanishes. The first layers receive a gradient that is a millionth of the original signal. They barely learn. This is why deep networks with sigmoid activations were so hard to train, even after backpropagation was discovered.

The opposite problem also exists: if weights are large, the chain of multiplications can grow exponentially. This is the exploding gradient problem. Gradients become enormous, weight updates become wild, and training diverges.

ReLU largely solved the vanishing gradient problem. For positive inputs, `relu'(z) = 1`. The gradient passes through without shrinking. A 10-layer network with ReLU activations and positive pre-activations has a gradient multiplier of `1^10 = 1`. No vanishing. The cost is dead neurons: if a neuron's pre-activation goes negative, its gradient is zero and it stops learning entirely.

### Weight Initialization

The other piece of the puzzle: how weights are initialized matters enormously for gradient flow. If initial weights are too large, activations saturate and gradients vanish (sigmoid) or explode. Too small, and signals die before reaching deeper layers.

[Xavier/Glorot initialization (2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) sets weights to maintain variance across layers with sigmoid or tanh activations:

```
w ~ Normal(0, sqrt(2 / (n_in + n_out)))
```

[He initialization (2015)](https://arxiv.org/abs/1502.01852) adjusts for ReLU, which zeros out half the distribution:

```
w ~ Normal(0, sqrt(2 / n_in))
```

Both strategies keep the variance of activations and gradients roughly constant as signals pass through the network. Without them, deep networks fail to train. With them, networks of 10, 50, or 100 layers become feasible.

<svg viewBox="0 0 480 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Vanishing gradient visualization showing a 5-layer deep network where the gradient magnitude shrinks exponentially from the output (large red bar) to the first hidden layer (tiny red bar), illustrating why early layers learn slowly with sigmoid activations." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Vanishing gradient: gradient magnitude shrinks exponentially from output to first layer in a sigmoid network</title>
  <rect width="480" height="180" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">vanishing gradient through 5 layers (sigmoid)</text>
  <!-- layers -->
  <rect x="30" y="40" width="60" height="80" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="60" y="85" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">layer 1</text>
  <rect x="110" y="40" width="60" height="80" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="140" y="85" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">layer 2</text>
  <rect x="190" y="40" width="60" height="80" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="220" y="85" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">layer 3</text>
  <rect x="270" y="40" width="60" height="80" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="300" y="85" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">layer 4</text>
  <rect x="350" y="40" width="60" height="80" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="380" y="85" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">output</text>
  <!-- gradient magnitude bars (bottom, growing right to left) -->
  <text x="240" y="140" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">gradient magnitude</text>
  <!-- layer 1: tiny gradient -->
  <rect x="45" y="150" width="2" height="12" rx="1" fill="#ef4444" opacity="0.4">
    <animate attributeName="opacity" values="0.4;0.4;1;0.4" keyTimes="0;0.8;0.9;1" dur="5s" repeatCount="indefinite"/>
  </rect>
  <text x="60" y="170" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">0.004</text>
  <!-- layer 2: small gradient -->
  <rect x="130" y="150" width="6" height="12" rx="1" fill="#ef4444" opacity="0.5">
    <animate attributeName="opacity" values="0.5;0.5;1;0.5" keyTimes="0;0.65;0.75;1" dur="5s" repeatCount="indefinite"/>
  </rect>
  <text x="140" y="170" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">0.016</text>
  <!-- layer 3: moderate gradient -->
  <rect x="207" y="150" width="16" height="12" rx="1" fill="#ef4444" opacity="0.6">
    <animate attributeName="opacity" values="0.6;0.6;1;0.6" keyTimes="0;0.50;0.60;1" dur="5s" repeatCount="indefinite"/>
  </rect>
  <text x="220" y="170" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">0.063</text>
  <!-- layer 4: larger gradient -->
  <rect x="280" y="148" width="30" height="14" rx="1" fill="#ef4444" opacity="0.7">
    <animate attributeName="opacity" values="0.7;0.7;1;0.7" keyTimes="0;0.35;0.45;1" dur="5s" repeatCount="indefinite"/>
  </rect>
  <text x="300" y="170" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">0.250</text>
  <!-- output: full gradient -->
  <rect x="355" y="146" width="50" height="16" rx="1" fill="#ef4444" opacity="0.9">
    <animate attributeName="opacity" values="0.9;0.9;1;0.9" keyTimes="0;0.20;0.30;1" dur="5s" repeatCount="indefinite"/>
  </rect>
  <text x="380" y="170" fill="#ef4444" font-family="monospace" font-size="6" text-anchor="middle">1.000</text>
  <!-- backward arrows -->
  <line x1="430" y1="80" x2="420" y2="80" stroke="#ef4444" stroke-width="1"/>
  <polygon points="422,77 416,80 422,83" fill="#ef4444"/>
  <text x="445" y="84" fill="#ef4444" font-family="monospace" font-size="7">dL/dz</text>
  <!-- annotation -->
  <text x="60" y="38" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">barely learns</text>
  <text x="380" y="38" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">learns well</text>
</svg>

*Figure 6: The vanishing gradient in a 5-layer sigmoid network. The gradient at the output is 1.0. After passing through each sigmoid derivative (max 0.25), it shrinks by at least 4x per layer. By layer 1, the gradient is 0.004, essentially zero. The first layers barely learn. ReLU fixes this: its derivative is 1 for positive inputs, so gradients pass through without shrinking.*

## Perceptron vs. MLP + Backpropagation

| Property | Perceptron (1958) | MLP + Backprop (1986) |
|---|---|---|
| Activation | Step function (binary) | Sigmoid, ReLU (continuous) |
| Learning rule | `w += eta * (t-y) * x` | `w -= eta * dL/dw` (chain rule) |
| Layers trainable | Output only | All layers |
| Decision boundary | Single hyperplane | Arbitrary nonlinear |
| Loss function | None (discrete error) | Cross-entropy, MSE |
| Credit assignment | Impossible for hidden layers | Solved by chain rule |
| XOR | Cannot learn | Learns automatically |

## Backpropagation in Code

The code follows the same progression from the previous posts. `neuron()` became `perceptron()` became `learn()` and `train()`. Now I replace the step function with sigmoid, replace the perceptron update rule with backpropagation, and the MLP learns from data.

```python
import math
import random

def sigmoid(z):
    """Logistic sigmoid: smooth, differentiable replacement for step."""
    return 1 / (1 + math.exp(max(-500, min(500, -z))))

def forward(x, weights, biases):
    """Forward pass. Returns list of activations for every layer."""
    activations = [list(x)]
    for W, b in zip(weights, biases):
        z = [sum(w * a for w, a in zip(row, activations[-1])) + bi
             for row, bi in zip(W, b)]
        activations.append([sigmoid(zi) for zi in z])
    return activations

def backward(activations, weights, target):
    """Backward pass. Returns weight and bias gradients for every layer."""
    output = activations[-1]
    delta = [o - t for o, t in zip(output, target)]
    w_grads, b_grads = [None] * len(weights), [None] * len(weights)
    for layer in reversed(range(len(weights))):
        a_prev = activations[layer]
        w_grads[layer] = [[d * a for a in a_prev] for d in delta]
        b_grads[layer] = list(delta)
        if layer > 0:
            a = activations[layer]
            delta = [sum(weights[layer][k][j] * delta[k]
                         for k in range(len(delta)))
                     * a[j] * (1 - a[j])
                     for j in range(len(a))]
    return w_grads, b_grads

def train_mlp(data, targets, sizes, eta=2.0, epochs=2000):
    """Train an MLP with backpropagation."""
    random.seed(42)
    weights, biases = [], []
    for i in range(len(sizes) - 1):
        s = (2 / (sizes[i] + sizes[i+1])) ** 0.5
        weights.append([[random.gauss(0, s) for _ in range(sizes[i])]
                        for _ in range(sizes[i+1])])
        biases.append([0.0] * sizes[i+1])
    for epoch in range(epochs):
        total_loss = 0
        for x, t in zip(data, targets):
            tv = [t] if isinstance(t, (int, float)) else t
            acts = forward(x, weights, biases)
            wg, bg = backward(acts, weights, tv)
            for l in range(len(weights)):
                for j in range(len(weights[l])):
                    for k in range(len(weights[l][j])):
                        weights[l][j][k] -= eta * wg[l][j][k]
                    biases[l][j] -= eta * bg[l][j]
            y = max(1e-15, min(1-1e-15, acts[-1][0]))
            total_loss += -(tv[0]*math.log(y) + (1-tv[0])*math.log(1-y))
        if epoch == 0 or (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1:4d}  loss: {total_loss/len(data):.4f}")
    return weights, biases
```

Training on XOR:

```python
data = [[0,0], [0,1], [1,0], [1,1]]
targets = [0, 1, 1, 0]
weights, biases = train_mlp(data, targets, sizes=[2, 2, 1])
```

```
Epoch    1  loss: 1.1687
Epoch  500  loss: 0.0085
Epoch 1000  loss: 0.0018
Epoch 1500  loss: 0.0010
Epoch 2000  loss: 0.0007
```

```python
for x, t in zip(data, targets):
    y = forward(x, weights, biases)[-1][0]
    print(f"  {x} -> {y:.4f}  (target: {t})")
```

```
  [0, 0] -> 0.0006  (target: 0)
  [0, 1] -> 0.9994  (target: 1)
  [1, 0] -> 0.9994  (target: 1)
  [1, 1] -> 0.0010  (target: 0)
```

XOR solved. Not hand-wired. Learned from data. The loss drops from 1.17 to 0.0007. The outputs are within a fraction of a percent of the targets. Compare this to the perceptron post, where `train()` ran for 1000 epochs on XOR and never converged, oscillating forever because a single hyperplane cannot separate the classes.

The difference is three functions: `sigmoid()` provides the smooth activation that makes calculus possible, `backward()` uses the chain rule to compute gradients for every weight in every layer, and `train_mlp()` uses those gradients to update all weights simultaneously. The credit assignment problem, the problem that killed Rosenblatt's research and froze neural networks for fifteen years, reduces to the chain rule applied to a composition of differentiable functions.

## Key Takeaways

I started where the perceptron post left off: a multi-layer architecture that could represent XOR but could not learn it. The gap was the credit assignment problem. Hidden neurons have no targets, so the perceptron learning rule cannot update their weights.

The solution required three changes. First, replace the step function with sigmoid, a smooth activation whose derivative exists everywhere and can be computed from its own output. Second, define cross-entropy as the loss function, giving a continuous, gradient-friendly measure of error that does not saturate when combined with sigmoid. Third, apply the chain rule to propagate the error backward through every layer. The transposed weight matrix `W2^T` distributes blame from the output to hidden neurons in proportion to their connection strength. The activation derivative gates the signal. The result is a gradient for every weight in every layer.

Backpropagation came with its own challenges. Sigmoid's maximum derivative of 0.25 causes gradients to vanish exponentially in deep networks. ReLU fixed that with a constant derivative of 1 for positive inputs. Xavier and He initialization kept gradients flowing through many layers. Adam adapted learning rates per parameter. Each fix enabled deeper networks, which enabled more powerful representations.

The progression so far: McCulloch-Pitts gave us a neuron that computes but cannot learn. Rosenblatt's perceptron added learning but was limited to linear boundaries. Backpropagation solved the credit assignment problem and enabled multi-layer networks to learn arbitrary nonlinear functions. But there is a missing capability. These networks process fixed-size inputs. Every example has to be the same shape: a vector of numbers with a predetermined length. Real-world data, language especially, does not work that way. A sentence can be three words or three hundred. The order matters. The meaning of a word depends on which words came before it. How do you handle sequences? That question drives the next post.
