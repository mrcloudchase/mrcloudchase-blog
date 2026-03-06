---
title: "The Perceptron: Rise, Fall, and Resurrection"
date: "2026-03-05"
excerpt: "The first machine that learned from data, the critique that nearly killed AI, and why Rosenblatt's insight still powers every neural network today."
author: "Chase Dovey"
tags: ["AI", "Deep Learning"]
draft: false
---

## Introduction

The previous post ended with a gap. The McCulloch-Pitts neuron could compute any Boolean function, but every threshold and every connection had to be set by hand. For three inputs and a simple logic gate, that is manageable. For a network that needs to recognize a face, read handwriting, or classify anything meaningful from raw data, hand-design is impossible. The model proved that neural computation works. It said nothing about how to make it learn.

In 1958, psychologist Frank Rosenblatt looked at the McCulloch-Pitts model and asked a natural follow-up question: what if the connections could adjust themselves? He took the same mathematical structure (inputs, summation, threshold, binary output) and modified it. He replaced the equal +1 excitatory contributions with variable real-valued weights. He replaced the fixed integer threshold with a learnable bias. And then he added the piece that McCulloch and Pitts never had: a learning rule. An algorithm that takes labeled examples, computes the error, and nudges the weights in the direction that would have produced the correct answer. He called the result the perceptron, described in his paper [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=65bebb15cdd2553d2af76f65b96d4e45b826094e).

The perceptron could learn. But it could only learn functions that a single hyperplane can separate. When Marvin Minsky and Seymour Papert proved this limitation in 1969, the obvious fix was to stack multiple perceptrons into layers, creating what is now called a multilayer perceptron (MLP). Rosenblatt himself proposed this architecture. The problem was that nobody, Rosenblatt included, could figure out how to train it. His learning rule only worked for a single layer. With no way to train deeper networks, the field stalled. Funding collapsed, researchers moved on, and neural networks entered a winter that lasted over a decade.

I wanted to put myself in Rosenblatt's shoes. He looked at McCulloch and Pitts' hand-designed neuron and asked: how do I make this learn? I want to ask myself the same question, starting from the M-P model I built in the previous post, and work through the reasoning that leads to the perceptron, its learning rule, its power, and its limits.

## From Counting to Weighting

I left the McCulloch-Pitts model with a clear limitation: every connection was equal and every threshold was fixed. If I want the model to learn from data, those fixed elements are the first things that need to change.

The M-P model treats all excitatory inputs equally: each active input contributes +1 to a running count, any active inhibitory input vetoes the output entirely, and the neuron fires if the count meets a fixed threshold. The connections have no strength. They are either on or off, excitatory or inhibitory, all equal.

I need to change three things.

**Variable real-valued weights.** Instead of every excitatory input contributing +1, what if each input gets its own connection strength? A real number that can be positive, negative, large, small, or zero. A weight of 3.2 means that input has strong excitatory influence. A weight of -1.5 means it has moderate inhibitory influence. A weight near zero means it barely matters. This replaces the binary excitatory/inhibitory distinction with a continuous spectrum of connection strengths.

**A bias term.** Instead of a fixed integer threshold, I replace it with a learnable bias that shifts the decision boundary. The threshold is no longer a separate parameter set by hand. It is absorbed into the bias and can be adjusted along with the weights.

**A learning rule.** This is the key piece. Instead of an engineer deciding what the weights should be, the model adjusts its own weights based on its mistakes. When it gets an answer wrong, it changes the weights in the direction that would have produced the correct answer. When it gets an answer right, it leaves the weights alone.

These three changes transform the McCulloch-Pitts neuron from a hand-designed logic gate into a machine that learns from data. This is exactly what Rosenblatt did. Let me build it step by step.

## The Perceptron Architecture

With variable weights and a bias, the model takes a vector of real-valued inputs, multiplies each by its corresponding weight, sums the results, adds the bias, and applies a step function:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
y = 1  if z >= 0
y = 0  if z < 0
```

In vector notation:

```
z = w . x + b
y = step(z)
```

The step function is the simplest possible activation: output 1 for non-negative inputs, 0 for negative inputs. It preserves the all-or-nothing firing behavior from biology, but now the decision of whether to fire depends on a weighted sum rather than a simple count. I have gone from counting active inputs to computing a weighted combination of them.

<svg viewBox="0 0 480 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Perceptron architecture showing inputs multiplied by variable weights, summed with a bias, passed through a step function to produce a binary output, with an error feedback loop that adjusts the weights during learning." style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Perceptron architecture with learning feedback loop</title>
  <rect width="480" height="190" rx="8" fill="#181818"/>
  <text x="240" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">perceptron architecture</text>
  <!-- inputs -->
  <circle cx="40" cy="45" r="13" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="49" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x1</text>
  <circle cx="40" cy="90" r="13" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="94" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">x2</text>
  <circle cx="40" cy="135" r="13" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="139" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">xn</text>
  <!-- weight edges -->
  <line x1="53" y1="45" x2="145" y2="80" stroke="#22d3ee" stroke-width="1.5"/>
  <line x1="53" y1="90" x2="145" y2="88" stroke="#22d3ee" stroke-width="1.5"/>
  <line x1="53" y1="135" x2="145" y2="96" stroke="#22d3ee" stroke-width="1.5"/>
  <!-- weight labels -->
  <text x="98" y="54" fill="#22d3ee" font-family="monospace" font-size="9" text-anchor="middle">w1</text>
  <text x="98" y="84" fill="#22d3ee" font-family="monospace" font-size="9" text-anchor="middle">w2</text>
  <text x="98" y="124" fill="#22d3ee" font-family="monospace" font-size="9" text-anchor="middle">wn</text>
  <!-- signal pulses along weight edges -->
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M53,45 L145,80" dur="3s" repeatCount="indefinite" keyTimes="0;0.25;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.23;0.25;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M53,90 L145,88" dur="3s" repeatCount="indefinite" keyTimes="0;0.25;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.23;0.25;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M53,135 L145,96" dur="3s" repeatCount="indefinite" keyTimes="0;0.25;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.23;0.25;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- summation + step node -->
  <circle cx="165" cy="88" r="22" fill="#333" stroke="#22d3ee" stroke-width="1.5">
    <animate attributeName="stroke" values="#22d3ee;#22d3ee;#fff;#fff;#22d3ee;#22d3ee" keyTimes="0;0.24;0.26;0.35;0.37;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="165" y="83" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">w.x+b</text>
  <text x="165" y="96" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">step</text>
  <!-- bias arrow -->
  <text x="165" y="52" fill="#fbbf24" font-family="monospace" font-size="9" text-anchor="middle">+b</text>
  <line x1="165" y1="56" x2="165" y2="66" stroke="#fbbf24" stroke-width="1" stroke-dasharray="2"/>
  <!-- output edge -->
  <line x1="187" y1="88" x2="235" y2="88" stroke="#555" stroke-width="1.5"/>
  <!-- output pulse -->
  <circle r="3" fill="#fbbf24" opacity="0">
    <animateMotion path="M187,88 L235,88" dur="3s" repeatCount="indefinite" keyTimes="0;0.36;0.46;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.35;0.36;0.44;0.46;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- output node -->
  <circle cx="250" cy="88" r="13" fill="#333" stroke="#fbbf24" stroke-width="1.5">
    <animate attributeName="stroke" values="#fbbf24;#fbbf24;#fff;#fff;#fbbf24;#fbbf24" keyTimes="0;0.45;0.47;0.52;0.54;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="250" y="92" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">y</text>
  <!-- compare box -->
  <line x1="263" y1="88" x2="300" y2="88" stroke="#555" stroke-width="1"/>
  <rect x="300" y="74" width="56" height="28" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="328" y="84" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">compare</text>
  <text x="328" y="96" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">y vs t</text>
  <!-- error label -->
  <line x1="356" y1="88" x2="390" y2="88" stroke="#ef4444" stroke-width="1"/>
  <text x="405" y="92" fill="#ef4444" font-family="monospace" font-size="9" text-anchor="start">error</text>
  <!-- feedback arrow -->
  <path d="M405,100 L405,170 L98,170 L98,90" fill="none" stroke="#ef4444" stroke-width="1" stroke-dasharray="4"/>
  <text x="250" y="165" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">adjust weights</text>
  <!-- animated feedback pulse -->
  <circle r="3" fill="#ef4444" opacity="0">
    <animateMotion path="M405,100 L405,170 L98,170 L98,90" dur="3s" repeatCount="indefinite" keyTimes="0;0.6;0.9;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.59;0.6;0.88;0.9;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- labels -->
  <text x="40" y="168" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">inputs</text>
  <text x="98" y="42" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">variable</text>
</svg>

*Figure 1: The perceptron architecture. Each input is multiplied by its own variable weight (cyan edges), summed with a bias, and passed through a step function. The output is compared to the target. When wrong, the error (red) feeds back to adjust the weights. This feedback loop is the learning mechanism that McCulloch-Pitts lacked.*

## The Learning Algorithm

Now I have an architecture with adjustable weights, but I still need a rule for adjusting them. The algorithm Rosenblatt devised is remarkably simple. Start with random weights. For each training example, compute the output. If the output is wrong, adjust the weights in the direction that would have produced the correct answer. If the output is right, do nothing. Repeat.

```
Initialize weights w and bias b to small random values
Set learning rate eta (typically 0.1 to 1.0)

For each training example (x, t) where t is the target:
    1. Compute output:  y = step(w . x + b)
    2. Compute error:   e = t - y
    3. Update weights:  w = w + eta * e * x
    4. Update bias:     b = b + eta * e

Repeat until all examples are classified correctly
```

When the output matches the target (`e = 0`), the weights do not change. When the output is wrong, one of two things happens:

**Missed positive** (`t = 1` but `y = 0`): The error is `e = 1`. The update adds `eta * x` to the weights. This makes the weighted sum larger for this input pattern on the next pass, pushing the output toward 1.

**False positive** (`t = 0` but `y = 1`): The error is `e = -1`. The update subtracts `eta * x` from the weights. This makes the weighted sum smaller for this input pattern, pushing the output toward 0.

The learning rate `eta` controls step size. Larger values mean faster convergence but risk overshooting. Smaller values converge more smoothly but take longer. For the perceptron, the convergence theorem guarantees convergence regardless of learning rate (as long as it is positive), so the choice affects speed, not correctness.

### Worked Example: Learning AND

I want to see this work concretely. The McCulloch-Pitts post required me to hand-design AND with a threshold of 2 and two equal +1 inputs. Now let me see if the perceptron can find a solution on its own.

Training data:

| x1 | x2 | target |
|-----|-----|--------|
| 0  | 0   | 0      |
| 0  | 1   | 0      |
| 1  | 0   | 0      |
| 1  | 1   | 1      |

Initialize: `w1 = 0, w2 = 0, b = 0, eta = 1`

**Epoch 1:**

```
(0,0) target 0:  z = 0*0 + 0*0 + 0 = 0   y = 1   e = -1
    w1 = 0 + (-1)*0 = 0    w2 = 0 + (-1)*0 = 0    b = 0 + (-1) = -1

(0,1) target 0:  z = 0*0 + 0*1 + (-1) = -1   y = 0   e = 0
    no update

(1,0) target 0:  z = 0*1 + 0*0 + (-1) = -1   y = 0   e = 0
    no update

(1,1) target 1:  z = 0*1 + 0*1 + (-1) = -1   y = 0   e = 1
    w1 = 0 + 1*1 = 1    w2 = 0 + 1*1 = 1    b = -1 + 1 = 0
```

After epoch 1: `w1 = 1, w2 = 1, b = 0`. Not converged yet, since (0,0) gives z = 0, which maps to y = 1 (false positive).

**Epoch 2:**

```
(0,0) target 0:  z = 1*0 + 1*0 + 0 = 0   y = 1   e = -1
    w1 = 1    w2 = 1    b = 0 + (-1) = -1

(0,1) target 0:  z = 1*0 + 1*1 + (-1) = 0   y = 1   e = -1
    w1 = 1    w2 = 1 + (-1)*1 = 0    b = -1 + (-1) = -2

(1,0) target 0:  z = 1*1 + 0*0 + (-2) = -1   y = 0   e = 0
    no update

(1,1) target 1:  z = 1*1 + 0*1 + (-2) = -1   y = 0   e = 1
    w1 = 1 + 1 = 2    w2 = 0 + 1 = 1    b = -2 + 1 = -1
```

After epoch 2: `w1 = 2, w2 = 1, b = -1`. Still not converged.

I will spare the remaining epochs. After a few more passes, the weights converge to a solution like `w1 = 2, w2 = 1, b = -2` (the exact values depend on the learning path). At that point:

```
(0,0): 2*0 + 1*0 - 2 = -2 < 0  -> y = 0  correct
(0,1): 2*0 + 1*1 - 2 = -1 < 0  -> y = 0  correct
(1,0): 2*1 + 1*0 - 2 =  0 >= 0 -> y = 1  ... wait, that gives 1, not 0
```

So that particular solution does not work. The algorithm keeps going. The key insight is that it *will* find a correct solution because AND is linearly separable. One valid solution is `w1 = 1, w2 = 1, b = -1.5`:

```
(0,0): 0 + 0 - 1.5 = -1.5 < 0  -> y = 0  correct
(0,1): 0 + 1 - 1.5 = -0.5 < 0  -> y = 0  correct
(1,0): 1 + 0 - 1.5 = -0.5 < 0  -> y = 0  correct
(1,1): 1 + 1 - 1.5 =  0.5 >= 0 -> y = 1  correct
```

The perceptron finds this (or an equivalent solution) automatically. No hand-design. That is the breakthrough.

## The Perceptron Convergence Theorem

The worked example shows the algorithm working, but does it always work? Rosenblatt claimed convergence for linearly separable data, and the formal proof was later provided by Novikoff in 1962:

**If the training data is linearly separable, the perceptron learning algorithm is guaranteed to converge to a correct solution in a finite number of steps.**

The proof sketch:

1. Assume a correct weight vector `w*` exists that correctly classifies all training data with some margin.
2. After each misclassification update, the dot product `w . w*` increases by at least a fixed positive amount (progress toward the solution).
3. After each update, the squared norm `||w||^2` increases by at most a bounded amount (the step size is limited).
4. Since `w . w*` grows linearly and `||w||` grows at most as the square root of the number of updates, the cosine of the angle between `w` and `w*` eventually reaches 1. But cosine is bounded by 1, so the number of updates must be finite.

The bound on the number of updates is:

```
number of updates <= (R / gamma)^2
```

Where `R` is the maximum norm of any training input and `gamma` is the margin of the optimal weight vector. Larger margin means fewer updates needed. Smaller margin means the algorithm has to work harder.

The critical condition is **linear separability**. If the data can be perfectly separated by a hyperplane (a line in 2D, a plane in 3D, a hyperplane in higher dimensions), the perceptron will find it. If the data is not linearly separable, the algorithm never converges. It oscillates forever, adjusting weights back and forth without settling. That condition will become very important shortly.

## Geometric Interpretation: Decision Boundaries

To understand what the convergence theorem is really saying, I need to think about what the perceptron computes geometrically.

The perceptron outputs `y = step(w . x + b)`. The boundary between "fire" (y = 1) and "don't fire" (y = 0) is the set of points where `w . x + b = 0`. In two dimensions, this equation defines a line. In three dimensions, a plane. In general, a **hyperplane**.

The weight vector `w` is perpendicular to this hyperplane. It points toward the "positive" side (where y = 1). The bias `b` shifts the hyperplane toward or away from the origin. So what the perceptron is really doing during training is searching for the orientation and position of a hyperplane that correctly separates the two classes.

<svg viewBox="0 0 460 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two-dimensional scatter plot showing class 0 points (red circles) and class 1 points (green filled circles) separated by an animated decision boundary line that rotates during training until it correctly separates the classes." style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Perceptron decision boundary in 2D: the boundary rotates during training until it separates the two classes</title>
  <rect width="460" height="250" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">decision boundary during training</text>
  <!-- axes -->
  <line x1="50" y1="220" x2="220" y2="220" stroke="#555" stroke-width="1"/>
  <line x1="50" y1="220" x2="50" y2="40" stroke="#555" stroke-width="1"/>
  <text x="135" y="240" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">x1</text>
  <text x="35" y="130" fill="#999" font-family="monospace" font-size="8" text-anchor="middle" transform="rotate(-90,35,130)">x2</text>
  <!-- class 0 points -->
  <circle cx="70" cy="195" r="6" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <circle cx="90" cy="175" r="6" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <circle cx="110" cy="190" r="6" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <circle cx="75" cy="165" r="6" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <circle cx="95" cy="200" r="6" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <!-- class 1 points -->
  <circle cx="150" cy="80" r="6" fill="#4ade80" stroke="#4ade80" stroke-width="1.5"/>
  <circle cx="170" cy="100" r="6" fill="#4ade80" stroke="#4ade80" stroke-width="1.5"/>
  <circle cx="180" cy="70" r="6" fill="#4ade80" stroke="#4ade80" stroke-width="1.5"/>
  <circle cx="160" cy="60" r="6" fill="#4ade80" stroke="#4ade80" stroke-width="1.5"/>
  <circle cx="190" cy="90" r="6" fill="#4ade80" stroke="#4ade80" stroke-width="1.5"/>
  <!-- decision boundary (animated rotation settling into correct position) -->
  <line x1="55" y1="100" x2="205" y2="200" stroke="#22d3ee" stroke-width="2">
    <animate attributeName="x1" values="140;110;80;55;55" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y1" values="35;55;80;100;100" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="x2" values="220;215;210;205;205" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="155;170;190;200;200" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
  </line>
  <!-- weight vector (perpendicular to boundary) -->
  <line x1="130" y1="150" x2="160" y2="120" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="3">
    <animate attributeName="x1" values="170;150;130;130;130" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y1" values="95;120;140;150;150" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="x2" values="190;170;160;160;160" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="70;95;110;120;120" keyTimes="0;0.3;0.6;0.85;1" dur="6s" repeatCount="indefinite"/>
  </line>
  <!-- legend -->
  <text x="290" y="55" fill="#999" font-family="monospace" font-size="9">Legend:</text>
  <circle cx="295" cy="72" r="5" fill="#4ade80"/>
  <text x="308" y="76" fill="#999" font-family="monospace" font-size="8">class 1 (y=1)</text>
  <circle cx="295" cy="92" r="5" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <text x="308" y="96" fill="#999" font-family="monospace" font-size="8">class 0 (y=0)</text>
  <line x1="288" y1="110" x2="302" y2="110" stroke="#22d3ee" stroke-width="2"/>
  <text x="308" y="114" fill="#999" font-family="monospace" font-size="8">decision boundary</text>
  <line x1="288" y1="128" x2="302" y2="128" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="3"/>
  <text x="308" y="132" fill="#999" font-family="monospace" font-size="8">weight vector w</text>
  <text x="350" y="180" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">the boundary rotates</text>
  <text x="350" y="193" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">with each weight update</text>
  <text x="350" y="206" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">until classes are</text>
  <text x="350" y="219" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">correctly separated</text>
</svg>

*Figure 2: The perceptron's decision boundary in 2D. Each weight update rotates the line. For a missed positive, the boundary rotates to include the missed point. For a false positive, it rotates to exclude it. The weight vector (yellow, dashed) is always perpendicular to the boundary, pointing toward the class-1 region. The convergence theorem guarantees that if a separating line exists, the algorithm will find it.*

Every time the perceptron makes an error, the weight update rotates and shifts the decision boundary. The convergence theorem says that if a correct boundary exists, the algorithm reaches it in finite steps. But what if no correct boundary exists? What if the problem requires a decision boundary that is not a straight line?

## The XOR Problem

This is where the perceptron breaks. Consider the XOR (exclusive or) function.

XOR outputs 1 when exactly one input is 1:

| x1 | x2 | XOR |
|-----|-----|-----|
| 0  | 0   | 0   |
| 0  | 1   | 1   |
| 1  | 0   | 1   |
| 1  | 1   | 0   |

In the McCulloch-Pitts post, I built XOR from a network of hand-designed neurons: AND, OR, and NOT composed together. A single M-P neuron could not compute it either. But now I have a learning rule. Can the perceptron learn XOR? The answer is no, and the proof is geometric. I can see it by plotting the four points:

<svg viewBox="0 0 460 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Side-by-side comparison of XOR data (not linearly separable, no single line can separate the classes) and AND data (linearly separable, a single line correctly separates the classes)." style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>XOR is not linearly separable: the class-1 points sit on opposite corners, making it impossible for a single line to separate them. AND is linearly separable.</title>
  <rect width="460" height="230" rx="8" fill="#181818"/>
  <text x="230" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">linear separability: XOR vs AND</text>
  <!-- LEFT: XOR data -->
  <text x="115" y="38" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9">XOR (not separable)</text>
  <line x1="40" y1="200" x2="200" y2="200" stroke="#555" stroke-width="1"/>
  <line x1="40" y1="200" x2="40" y2="50" stroke="#555" stroke-width="1"/>
  <text x="120" y="218" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">x1</text>
  <text x="28" y="125" fill="#999" font-family="monospace" font-size="8" text-anchor="middle" transform="rotate(-90,28,125)">x2</text>
  <!-- (0,0) = 0 -->
  <circle cx="60" cy="185" r="8" fill="none" stroke="#ef4444" stroke-width="2"/>
  <text x="60" y="189" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">0</text>
  <!-- (1,0) = 1 -->
  <circle cx="180" cy="185" r="8" fill="#4ade80" stroke="#4ade80" stroke-width="2"/>
  <text x="180" y="189" fill="#181818" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <!-- (0,1) = 1 -->
  <circle cx="60" cy="70" r="8" fill="#4ade80" stroke="#4ade80" stroke-width="2"/>
  <text x="60" y="74" fill="#181818" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <!-- (1,1) = 0 -->
  <circle cx="180" cy="70" r="8" fill="none" stroke="#ef4444" stroke-width="2"/>
  <text x="180" y="74" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">0</text>
  <!-- animated line attempts (all fail) -->
  <line x1="35" y1="130" x2="200" y2="130" stroke="#555" stroke-width="1" stroke-dasharray="4">
    <animate attributeName="y1" values="130;60;180;130" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="130;180;60;130" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="stroke" values="#555;#ef4444;#ef4444;#555" keyTimes="0;0.1;0.9;1" dur="6s" repeatCount="indefinite"/>
  </line>
  <!-- RIGHT: AND data -->
  <text x="350" y="38" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9">AND (separable)</text>
  <line x1="270" y1="200" x2="430" y2="200" stroke="#555" stroke-width="1"/>
  <line x1="270" y1="200" x2="270" y2="50" stroke="#555" stroke-width="1"/>
  <text x="350" y="218" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">x1</text>
  <!-- (0,0) = 0 -->
  <circle cx="290" cy="185" r="8" fill="none" stroke="#ef4444" stroke-width="2"/>
  <text x="290" y="189" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">0</text>
  <!-- (1,0) = 0 -->
  <circle cx="410" cy="185" r="8" fill="none" stroke="#ef4444" stroke-width="2"/>
  <text x="410" y="189" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">0</text>
  <!-- (0,1) = 0 -->
  <circle cx="290" cy="70" r="8" fill="none" stroke="#ef4444" stroke-width="2"/>
  <text x="290" y="74" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">0</text>
  <!-- (1,1) = 1 -->
  <circle cx="410" cy="70" r="8" fill="#4ade80" stroke="#4ade80" stroke-width="2"/>
  <text x="410" y="74" fill="#181818" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <!-- separating line (works) -->
  <line x1="370" y1="45" x2="435" y2="200" stroke="#4ade80" stroke-width="2"/>
</svg>

*Figure 3: XOR vs AND. The AND function is linearly separable: a single line cleanly divides the class-1 point (top right) from the class-0 points. XOR is not: the class-1 points (top left, bottom right) sit on opposite corners, and no single line can separate them from the class-0 points (bottom left, top right). The dashed line rotates to show that every possible orientation misclassifies at least one point.*

The class-1 points (0,1) and (1,0) sit on opposite corners. The class-0 points (0,0) and (1,1) sit on the other two corners. No single straight line can put the 1s on one side and the 0s on the other.

The algebraic proof is equally clean. For the perceptron to produce the correct outputs, I need:

```
(0,0) -> 0:  w1*0 + w2*0 + b < 0       =>  b < 0
(0,1) -> 1:  w1*0 + w2*1 + b >= 0       =>  w2 >= -b
(1,0) -> 1:  w1*1 + w2*0 + b >= 0       =>  w1 >= -b
(1,1) -> 0:  w1*1 + w2*1 + b < 0        =>  w1 + w2 < -b
```

From constraints 2 and 3: `w1 >= -b` and `w2 >= -b`. Adding these: `w1 + w2 >= -2b`. Since `b < 0`, we know `-b > 0`, so `-2b > 0`, and therefore `w1 + w2 > 0`.

But constraint 4 says `w1 + w2 < -b`. Since `-b > 0`, this says `w1 + w2` is positive, which is consistent so far. The contradiction: `w1 + w2 >= -2b` (from constraints 2+3) but also `w1 + w2 < -b` (constraint 4). This requires `-2b <= w1 + w2 < -b`, which means `-2b < -b`, which means `-b < 0`, which means `b > 0`. But constraint 1 says `b < 0`. Contradiction.

No values of `w1`, `w2`, and `b` satisfy all four constraints simultaneously. A single perceptron cannot compute XOR.

## The Obvious Fix and the Real Problem

The fix for XOR is conceptually obvious to me: add another layer of neurons. In the previous post, I built XOR from a network of McCulloch-Pitts neurons by composing AND, OR, and NOT gates. The same idea applies here. If I stack perceptrons into layers, with the outputs of one layer feeding as inputs to the next, the network can learn nonlinear decision boundaries. This architecture, multiple layers of perceptrons, is called a **multilayer perceptron** (MLP).

Rosenblatt knew this too. He proposed multi-layer architectures and understood that they could solve problems beyond linear separability. The issue was not the architecture. The issue was **training**. The perceptron learning rule works by comparing the output to the target and adjusting the weights accordingly. But in a multi-layer network, how do I adjust the weights in the first layer? Those neurons do not have targets. They produce intermediate representations that feed into the next layer, and there is no direct way to know what those intermediate values should be. The learning rule has no mechanism for assigning credit (or blame) to neurons that are not directly connected to the output.

This is the **credit assignment problem**. I can build multi-layer networks. I cannot train them. And neither could Rosenblatt.

## Minsky, Papert, and the AI Winter

In 1969, Marvin Minsky and Seymour Papert published *Perceptrons*, a rigorous mathematical analysis of what single-layer perceptrons can and cannot compute. Their analysis went beyond XOR. They showed that single-layer perceptrons cannot compute:

- **Parity functions**: determining whether an odd or even number of inputs are active (XOR is the 2-input case)
- **Connectedness**: determining whether a pattern on a grid forms a single connected region
- **Symmetry detection**: determining whether a pattern is symmetric

These are not exotic edge cases. Many real-world classification problems require detecting relationships between inputs that cannot be captured by a single hyperplane.

Their book was careful to note that multi-layer networks could solve these problems. But they expressed skepticism that anyone would find a practical learning algorithm for them:

> "The perceptron has shown itself worthy of study despite (and even because of) its severe limitations. It has many features to attract attention: its linearity; its intriguing learning theorem; its clear paradigmatic simplicity as a kind of parallel computation. There is no reason to suppose that any of these virtues carry over to the many-layered version."

That skepticism landed on top of an already unsolved problem. Rosenblatt had no multi-layer learning algorithm. Minsky and Papert had just proved the single-layer model was fundamentally limited. The combination was devastating.

Research funding for neural networks collapsed. The U.S. government and military, which had been significant funders of perceptron research, redirected AI spending toward symbolic AI: rule-based systems, expert systems, logical reasoning. Academic researchers moved on. PhD students were advised to avoid neural networks. The period from roughly 1969 to the mid-1980s is known as the neural network winter.

The cause was not XOR itself. XOR was just the clearest demonstration of a deeper problem: the perceptron learning rule could not train multi-layer networks, and single-layer networks could not solve interesting problems. The field was stuck between an architecture that was too simple and a learning algorithm that could not scale.

The irony is that the solution was already emerging. Paul Werbos described backpropagation in his 1974 PhD thesis, providing exactly the multi-layer learning algorithm that solved the credit assignment problem. But Werbos's work went largely unnoticed. It took until 1986, when David Rumelhart, Geoffrey Hinton, and Ronald Williams published their landmark paper demonstrating backpropagation on multi-layer networks, for the field to revive.

The perceptron was vindicated, not as a complete solution, but as the foundation. Everything that came after builds on the principles Rosenblatt introduced: variable weights, learned representations, error-driven updates. The missing piece was a way to propagate that error backward through multiple layers. That is the next post.

## McCulloch-Pitts vs. Perceptron: What Changed

| Property | McCulloch-Pitts (1943) | Perceptron (1958) |
|---|---|---|
| Connection strength | Equal (+1 excitatory, absolute inhibitory veto) | Variable real-valued weights |
| Threshold | Fixed integer, hand-set | Learnable bias term |
| Learning | None (hand-designed) | Perceptron learning rule |
| Input types | Binary (0 or 1) | Real-valued |
| Inhibition | Absolute veto (any active inhibitory input blocks firing) | Negative weight (graded inhibition) |
| Convergence guarantee | N/A | Yes, for linearly separable data |
| Limitation | No learning | Only linearly separable problems |

Starting from the McCulloch-Pitts model, I kept the core structure (inputs, summation, threshold activation, binary output) and replaced every fixed element with a learnable one. That single conceptual shift, from hand-designed to data-driven, is the dividing line between computation and learning. It is exactly what Rosenblatt did.

## The Perceptron in Code

Everything I have built reduces to a short Python implementation. The function below trains and predicts:

```python
def perceptron_train(data, targets, eta=1.0, max_epochs=100):
    """Train a perceptron. Returns learned weights and bias."""
    n_features = len(data[0])
    w = [0.0] * n_features
    b = 0.0

    for epoch in range(max_epochs):
        errors = 0
        for x, t in zip(data, targets):
            z = sum(wi * xi for wi, xi in zip(w, x)) + b
            y = 1 if z >= 0 else 0
            e = t - y
            if e != 0:
                errors += 1
                w = [wi + eta * e * xi for wi, xi in zip(w, x)]
                b = b + eta * e
        if errors == 0:
            print(f"Converged after {epoch + 1} epochs")
            break
    return w, b


def perceptron_predict(x, w, b):
    """Predict with a trained perceptron."""
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    return 1 if z >= 0 else 0
```

Training on AND:

```python
data = [[0,0], [0,1], [1,0], [1,1]]
w, b = perceptron_train(data, targets=[0, 0, 0, 1])
```

```
Converged after 6 epochs
```

```python
for x in data:
    print(f"  {x} -> {perceptron_predict(x, w, b)}")
```

```
  [0, 0] -> 0
  [0, 1] -> 0
  [1, 0] -> 0
  [1, 1] -> 1
```

OR converges equally fast:

```python
w, b = perceptron_train(data, targets=[0, 1, 1, 1])
```

```
Converged after 4 epochs
```

XOR never converges:

```python
w, b = perceptron_train(data, targets=[0, 1, 1, 0], max_epochs=1000)
```

```
(no convergence message -- the loop runs all 1000 epochs)
```

```python
for x in data:
    print(f"  {x} -> {perceptron_predict(x, w, b)}")
```

```
  [0, 0] -> 1
  [0, 1] -> 1
  [1, 0] -> 0
  [1, 1] -> 0
```

At least one point is always wrong. The algorithm oscillates, adjusting the weights to fix one error only to create another. This is the convergence theorem in reverse: XOR is not linearly separable, so the algorithm cannot settle.

## Key Takeaways

I started where the McCulloch-Pitts post left off: a model that computes but cannot learn. Following Rosenblatt's reasoning, I replaced equal contributions with variable weights, the fixed threshold with a learnable bias, and hand-design with an error-driven update rule. The convergence theorem guaranteed that if a solution exists (if the data is linearly separable), the algorithm finds it. But linear separability is a hard constraint. XOR, parity, connectedness, and most real-world problems require decision boundaries that a single hyperplane cannot express. The fix, stacking perceptrons into multiple layers, was architecturally obvious. The problem was training: Rosenblatt's learning rule could not assign credit to neurons that were not directly connected to the output. That unsolved problem froze the field for over a decade. The solution, a way to propagate error backward through every layer of a deep network, is the subject of the next post.
