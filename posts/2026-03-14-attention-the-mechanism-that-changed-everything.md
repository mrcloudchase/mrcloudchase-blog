---
title: "Attention: The Mechanism That Changed Everything"
date: "2026-03-14"
excerpt: "Why sequential models hit a wall with long sequences, and how the attention mechanism - queries, keys, and values - solved the problem and unlocked modern AI."
author: "Chase Dovey"
tags: ["AI", "Deep Learning"]
draft: true
---

## Introduction

MLPs are powerful function approximators, but they have a fundamental limitation: they operate on fixed-size inputs. Feed an MLP a sentence of 10 words, it produces an output. Feed it 11 words, it breaks. Real-world sequences, text, audio, time series, vary in length. An architecture that handles sequences needs to process variable-length inputs and, crucially, understand the **relationships between elements** at different positions.

Recurrent Neural Networks (RNNs) solved the variable-length problem by processing sequences one element at a time, maintaining a hidden state that summarizes everything seen so far. But this sequential bottleneck introduced its own problems: vanishing gradients over long sequences, inability to parallelize, and a hidden state that had to compress an entire sequence into a single fixed-size vector.

The attention mechanism solved all of these problems. Instead of compressing the entire sequence into one vector, attention lets the model look back at every previous element and decide, dynamically, which elements are relevant for the current computation. This is the single most important architectural innovation in the history of deep learning.

## The Sequence Problem

Consider translating a sentence from English to French. An MLP takes a fixed-size input, so you would need to pad or truncate every sentence to the same length. Worse, an MLP treats each input position independently, so it has no way to learn that "the" in position 1 relates to "cat" in position 3.

Sequences have structure that matters:
- **Order matters**: "dog bites man" is different from "man bites dog"
- **Long-range dependencies**: In "The cat that sat on the mat was orange," the verb "was" depends on "cat" across 6 intervening words
- **Variable length**: Sentences range from 1 word to hundreds

## Recurrent Neural Networks (RNNs)

RNNs process sequences by maintaining a **hidden state** that gets updated at each time step:

```
h_t = activation(W_h . h_{t-1} + W_x . x_t + b)
y_t = W_y . h_t + b_y
```

At each step, the RNN takes the current input `x_t` and the previous hidden state `h_{t-1}`, combines them, and produces a new hidden state `h_t`. The hidden state is the network's memory of everything it has seen so far.

<svg viewBox="0 0 480 170" xmlns="http://www.w3.org/2000/svg" style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="480" height="170" rx="8" fill="#181818"/>
  <text x="240" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">RNN unrolled through time</text>
  <!-- time step boxes -->
  <rect x="30" y="60" width="60" height="40" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="60" y="84" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">RNN</text>
  <rect x="140" y="60" width="60" height="40" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="170" y="84" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">RNN</text>
  <rect x="250" y="60" width="60" height="40" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="280" y="84" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">RNN</text>
  <rect x="360" y="60" width="60" height="40" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="390" y="84" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">RNN</text>
  <!-- inputs from below -->
  <text x="60" y="148" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">x1</text>
  <line x1="60" y1="138" x2="60" y2="100" stroke="#4ade80" stroke-width="1"/>
  <text x="170" y="148" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">x2</text>
  <line x1="170" y1="138" x2="170" y2="100" stroke="#4ade80" stroke-width="1"/>
  <text x="280" y="148" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">x3</text>
  <line x1="280" y1="138" x2="280" y2="100" stroke="#4ade80" stroke-width="1"/>
  <text x="390" y="148" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">x4</text>
  <line x1="390" y1="138" x2="390" y2="100" stroke="#4ade80" stroke-width="1"/>
  <!-- hidden state arrows between boxes -->
  <line x1="90" y1="80" x2="140" y2="80" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="115" y="74" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">h1</text>
  <line x1="200" y1="80" x2="250" y2="80" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="225" y="74" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">h2</text>
  <line x1="310" y1="80" x2="360" y2="80" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="335" y="74" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">h3</text>
  <line x1="420" y1="80" x2="460" y2="80" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="445" y="74" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">h4</text>
  <!-- outputs above -->
  <line x1="60" y1="60" x2="60" y2="40" stroke="#888" stroke-width="1"/>
  <text x="60" y="36" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">y1</text>
  <line x1="170" y1="60" x2="170" y2="40" stroke="#888" stroke-width="1"/>
  <text x="170" y="36" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">y2</text>
  <line x1="280" y1="60" x2="280" y2="40" stroke="#888" stroke-width="1"/>
  <text x="280" y="36" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">y3</text>
  <line x1="390" y1="60" x2="390" y2="40" stroke="#888" stroke-width="1"/>
  <text x="390" y="36" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">y4</text>
  <!-- gradient fade visualization -->
  <rect x="30" y="105" width="60" height="4" rx="2" fill="#ef4444" opacity="0.2"/>
  <rect x="140" y="105" width="60" height="4" rx="2" fill="#ef4444" opacity="0.4"/>
  <rect x="250" y="105" width="60" height="4" rx="2" fill="#ef4444" opacity="0.7"/>
  <rect x="360" y="105" width="60" height="4" rx="2" fill="#ef4444" opacity="1.0"/>
  <text x="240" y="122" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">gradient strength (fades backward through time)</text>
  <!-- animated signal -->
  <circle r="3" fill="#fbbf24" opacity="0">
    <animateMotion path="M60,80 L115,80 L170,80 L225,80 L280,80 L335,80 L390,80 L445,80" dur="4s" repeatCount="indefinite" keyTimes="0;0.05;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.83;0.85;1" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>

### The Vanishing Gradient Problem in RNNs

Training an RNN means backpropagating through time (BPTT). The gradient at time step 1 requires multiplying gradients through every intermediate step. For a sequence of length T, the gradient is a product of T Jacobian matrices. If the largest eigenvalue of these matrices is less than 1, the product shrinks exponentially. If it is greater than 1, it explodes.

For long sequences (T > 20-50), the gradient signal from the end of the sequence to the beginning is effectively zero. The RNN cannot learn long-range dependencies. It forgets.

LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) mitigate this with gating mechanisms that control information flow. But they do not eliminate the fundamental problem: processing is sequential. Each step depends on the previous step, so the computation cannot be parallelized. Training on long sequences is slow.

## The Bottleneck Problem

In sequence-to-sequence tasks (like translation), the standard RNN approach was the **encoder-decoder** architecture:

1. An encoder RNN processes the input sequence, compressing it into a final hidden state (the "context vector")
2. A decoder RNN generates the output sequence from this context vector

The problem: the entire input sequence, regardless of length, must be compressed into a single fixed-size vector. For short sequences, this works. For long sequences, information is inevitably lost. The context vector becomes a bottleneck.

<svg viewBox="0 0 460 160" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="160" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">the bottleneck problem</text>
  <!-- encoder -->
  <rect x="20" y="55" width="40" height="30" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="40" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">enc</text>
  <rect x="68" y="55" width="40" height="30" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="88" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">enc</text>
  <rect x="116" y="55" width="40" height="30" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="136" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">enc</text>
  <rect x="164" y="55" width="40" height="30" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="184" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">enc</text>
  <!-- arrows between encoder steps -->
  <line x1="60" y1="70" x2="68" y2="70" stroke="#555" stroke-width="1"/>
  <line x1="108" y1="70" x2="116" y2="70" stroke="#555" stroke-width="1"/>
  <line x1="156" y1="70" x2="164" y2="70" stroke="#555" stroke-width="1"/>
  <!-- encoder inputs -->
  <text x="40" y="100" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">The</text>
  <text x="88" y="100" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">cat</text>
  <text x="136" y="100" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">sat</text>
  <text x="184" y="100" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">down</text>
  <!-- bottleneck -->
  <line x1="204" y1="70" x2="228" y2="70" stroke="#fbbf24" stroke-width="2"/>
  <circle cx="240" cy="70" r="12" fill="#333" stroke="#fbbf24" stroke-width="2"/>
  <text x="240" y="74" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">ctx</text>
  <text x="240" y="40" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">bottleneck!</text>
  <text x="240" y="50" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">entire sequence</text>
  <text x="240" y="58" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">in one vector</text>
  <line x1="252" y1="70" x2="268" y2="70" stroke="#fbbf24" stroke-width="2"/>
  <!-- decoder -->
  <rect x="268" y="55" width="40" height="30" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="288" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">dec</text>
  <rect x="316" y="55" width="40" height="30" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="336" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">dec</text>
  <rect x="364" y="55" width="40" height="30" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="384" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">dec</text>
  <rect x="412" y="55" width="40" height="30" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="432" y="74" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">dec</text>
  <!-- arrows between decoder steps -->
  <line x1="308" y1="70" x2="316" y2="70" stroke="#555" stroke-width="1"/>
  <line x1="356" y1="70" x2="364" y2="70" stroke="#555" stroke-width="1"/>
  <line x1="404" y1="70" x2="412" y2="70" stroke="#555" stroke-width="1"/>
  <!-- decoder outputs -->
  <text x="288" y="100" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Le</text>
  <text x="336" y="100" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">chat</text>
  <text x="384" y="100" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">s'est</text>
  <text x="432" y="100" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">assis</text>
  <!-- labels -->
  <text x="112" y="140" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">encoder</text>
  <text x="360" y="140" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">decoder</text>
</svg>

## Bahdanau Attention (2014)

Dzmitry Bahdanau proposed a solution in 2014: instead of compressing the input into a single vector, let the decoder look at **all** encoder hidden states and learn which ones are relevant for each output step.

The idea: at each decoder step, compute a relevance score between the current decoder state and every encoder state. Use these scores as weights to create a weighted combination of encoder states. This weighted combination is the context for the current step.

```
score(s_t, h_i) = v^T . tanh(W_s . s_t + W_h . h_i)   # alignment score
alpha_i = softmax(score(s_t, h_i))                       # attention weights
context = sum(alpha_i * h_i)                              # weighted sum of encoder states
```

Where `s_t` is the decoder state at step `t`, `h_i` are the encoder states, and `alpha_i` are the attention weights.

The attention weights `alpha_i` form a probability distribution over the input sequence. They tell the decoder: "for this output word, focus mostly on these input words." The model learns these weights during training.

## Self-Attention: Attending to Yourself

Bahdanau attention lets a decoder attend to an encoder. **Self-attention** generalizes this: every position in a sequence attends to every other position in the same sequence.

This is the key insight of the transformer. Instead of processing the sequence step by step (like an RNN), self-attention processes all positions simultaneously. Each position directly sees every other position. No information bottleneck. No vanishing gradients through time. Full parallelization.

## The QKV Formulation

The transformer's attention mechanism uses three learned projections of each input: **queries (Q)**, **keys (K)**, and **values (V)**.

The analogy: think of a database lookup. You have a **query** (what you are looking for), a set of **keys** (labels for the stored items), and **values** (the stored items themselves). The attention mechanism scores how well each key matches the query, then returns a weighted sum of the values.

Given an input matrix `X` (one row per position, one column per feature):

```
Q = X . W_Q    # queries: "what am I looking for?"
K = X . W_K    # keys: "what do I contain?"
V = X . W_V    # values: "what do I provide?"
```

The attention computation:

```
Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_k)) . V
```

Step by step:

1. **Compute scores**: `Q . K^T` produces a matrix of dot products between every query and every key. Entry (i, j) measures how much position i should attend to position j.
2. **Scale**: Divide by `sqrt(d_k)` where `d_k` is the dimension of the keys. Without scaling, for large `d_k` the dot products become large, pushing softmax into regions with tiny gradients.
3. **Normalize**: Apply softmax row-wise, converting scores to probabilities. Each row sums to 1.
4. **Aggregate**: Multiply by V, producing a weighted sum of values for each position.

<svg viewBox="0 0 480 220" xmlns="http://www.w3.org/2000/svg" style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="480" height="220" rx="8" fill="#181818"/>
  <text x="240" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">scaled dot-product attention</text>
  <!-- input X -->
  <rect x="20" y="80" width="40" height="50" rx="4" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="109" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">X</text>
  <!-- three projection arrows -->
  <line x1="60" y1="90" x2="100" y2="60" stroke="#22d3ee" stroke-width="1"/>
  <line x1="60" y1="105" x2="100" y2="105" stroke="#fbbf24" stroke-width="1"/>
  <line x1="60" y1="120" x2="100" y2="150" stroke="#4ade80" stroke-width="1"/>
  <!-- Q, K, V boxes -->
  <rect x="100" y="42" width="40" height="35" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="120" y="64" fill="#22d3ee" font-family="monospace" font-size="10" text-anchor="middle">Q</text>
  <rect x="100" y="90" width="40" height="35" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="120" y="112" fill="#fbbf24" font-family="monospace" font-size="10" text-anchor="middle">K</text>
  <rect x="100" y="138" width="40" height="35" rx="4" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="120" y="160" fill="#4ade80" font-family="monospace" font-size="10" text-anchor="middle">V</text>
  <!-- Q.K^T -->
  <line x1="140" y1="59" x2="180" y2="80" stroke="#22d3ee" stroke-width="1"/>
  <line x1="140" y1="107" x2="180" y2="90" stroke="#fbbf24" stroke-width="1"/>
  <rect x="180" y="68" width="55" height="35" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="208" y="82" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">Q . K^T</text>
  <text x="208" y="95" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">/ sqrt(dk)</text>
  <!-- softmax -->
  <line x1="235" y1="85" x2="260" y2="85" stroke="#888" stroke-width="1"/>
  <rect x="260" y="68" width="50" height="35" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="285" y="90" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">softmax</text>
  <!-- attention weights matrix -->
  <line x1="310" y1="85" x2="335" y2="85" stroke="#888" stroke-width="1"/>
  <rect x="335" y="68" width="35" height="35" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="352" y="90" fill="#ef4444" font-family="monospace" font-size="9" text-anchor="middle">A</text>
  <!-- multiply by V -->
  <line x1="370" y1="85" x2="390" y2="110" stroke="#ef4444" stroke-width="1"/>
  <line x1="140" y1="155" x2="390" y2="120" stroke="#4ade80" stroke-width="1"/>
  <rect x="390" y="100" width="55" height="35" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="418" y="122" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">A . V</text>
  <!-- output -->
  <line x1="445" y1="117" x2="465" y2="117" stroke="#888" stroke-width="1"/>
  <text x="470" y="121" fill="#fbbf24" font-family="monospace" font-size="9" text-anchor="start">out</text>
  <!-- labels -->
  <text x="75" y="50" fill="#22d3ee" font-family="monospace" font-size="7">x W_Q</text>
  <text x="75" y="102" fill="#fbbf24" font-family="monospace" font-size="7">x W_K</text>
  <text x="75" y="165" fill="#4ade80" font-family="monospace" font-size="7">x W_V</text>
  <text x="352" y="55" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">attention</text>
  <text x="352" y="63" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">weights</text>
  <!-- animated flow -->
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M40,105 L120,59 L208,85 L285,85 L352,85 L418,117" dur="4s" repeatCount="indefinite" keyTimes="0;0.05;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.83;0.85;1" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>

### Why sqrt(d_k)?

Without the scaling factor, the dot products `Q . K^T` grow proportionally to `d_k`. When `d_k = 512` (a typical value), the dot products can be in the hundreds. Softmax of large values produces near-one-hot distributions with near-zero gradients. Dividing by `sqrt(d_k)` keeps the variance of the scores around 1, maintaining healthy gradients.

## Multi-Head Attention

A single attention head can only learn one type of relationship. "Multi-head attention" runs multiple attention heads in parallel, each with its own Q, K, V projections, then concatenates and projects the results:

```
head_i = Attention(X . W_Q_i, X . W_K_i, X . W_V_i)
MultiHead(X) = Concat(head_1, ..., head_h) . W_O
```

With 8 heads (typical for base models) and `d_model = 512`, each head works with `d_k = 512/8 = 64` dimensions. Different heads learn different types of attention patterns:
- One head might attend to the previous word (syntactic adjacency)
- Another might attend to the subject of the sentence (semantic role)
- Another might attend to the verb (predicate structure)
- Another might learn positional patterns

The concatenation and output projection `W_O` combine these different views into a unified representation.

<svg viewBox="0 0 460 180" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="180" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">multi-head attention</text>
  <!-- input -->
  <rect x="20" y="70" width="40" height="40" rx="4" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="94" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">X</text>
  <!-- split arrows -->
  <line x1="60" y1="78" x2="95" y2="45" stroke="#555" stroke-width="1"/>
  <line x1="60" y1="85" x2="95" y2="72" stroke="#555" stroke-width="1"/>
  <line x1="60" y1="92" x2="95" y2="100" stroke="#555" stroke-width="1"/>
  <line x1="60" y1="99" x2="95" y2="128" stroke="#555" stroke-width="1"/>
  <!-- attention heads -->
  <rect x="95" y="30" width="70" height="28" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="130" y="48" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">head 1</text>
  <rect x="95" y="62" width="70" height="28" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="130" y="80" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">head 2</text>
  <rect x="95" y="94" width="70" height="28" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="130" y="112" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">head 3</text>
  <rect x="95" y="126" width="70" height="28" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="130" y="144" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">head h</text>
  <!-- concat arrows -->
  <line x1="165" y1="44" x2="210" y2="72" stroke="#555" stroke-width="1"/>
  <line x1="165" y1="76" x2="210" y2="80" stroke="#555" stroke-width="1"/>
  <line x1="165" y1="108" x2="210" y2="88" stroke="#555" stroke-width="1"/>
  <line x1="165" y1="140" x2="210" y2="96" stroke="#555" stroke-width="1"/>
  <!-- concat box -->
  <rect x="210" y="62" width="55" height="42" rx="4" fill="#333" stroke="#888" stroke-width="1.5"/>
  <text x="238" y="87" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">concat</text>
  <!-- W_O projection -->
  <line x1="265" y1="83" x2="295" y2="83" stroke="#555" stroke-width="1.5"/>
  <rect x="295" y="68" width="45" height="30" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="318" y="87" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">W_O</text>
  <!-- output -->
  <line x1="340" y1="83" x2="370" y2="83" stroke="#555" stroke-width="1.5"/>
  <rect x="370" y="68" width="55" height="30" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="398" y="87" fill="#fbbf24" font-family="monospace" font-size="9" text-anchor="middle">output</text>
  <!-- animated pulses through heads -->
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M60,78 L130,44 L210,72 L238,83 L318,83 L398,83" dur="3s" repeatCount="indefinite" keyTimes="0;0.1;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.1;0.83;0.85;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- dimension annotation -->
  <text x="130" y="170" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">each head: d_k = d_model / h</text>
  <text x="340" y="170" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">output: d_model</text>
</svg>

## What Attention Actually Learns

Attention patterns in trained models reveal interpretable structure. In language models:

- **Syntactic heads**: Attend to the syntactic parent/child of each word. Verbs attend to their subjects.
- **Positional heads**: Attend to the previous or next position, implementing local context.
- **Rare word heads**: Attend backward to infrequent tokens that carry high information content.
- **Separator heads**: Attend to punctuation and sentence boundaries.
- **Induction heads**: Detect repeated patterns. If "A B ... A" appears, attend from the second A back to B to predict what comes next.

These patterns emerge from training, not from any explicit programming. The network discovers that these attention strategies are useful for predicting the next token.

## Computational Complexity

Self-attention has a critical trade-off. Computing `Q . K^T` produces an `n x n` matrix where `n` is the sequence length. This means:

- **Time complexity**: O(n^2 * d) per layer
- **Memory complexity**: O(n^2) for storing the attention matrix

For a sequence of 1000 tokens, the attention matrix has 1,000,000 entries. For 10,000 tokens, 100,000,000 entries. This quadratic scaling is the primary bottleneck for processing long sequences.

Various approaches address this:
- **Sparse attention**: Only compute attention for a subset of position pairs
- **Linear attention**: Approximate the softmax with kernel functions to achieve O(n) complexity
- **Flash Attention**: Exact attention computed in a memory-efficient way using tiling and recomputation
- **Sliding window attention**: Only attend within a fixed-size window around each position

## Key Takeaways

RNNs process sequences step by step and suffer from vanishing gradients over long sequences. The encoder-decoder bottleneck forces the entire input into a single vector. Attention eliminates both problems by letting every position directly access every other position. The QKV formulation provides a learnable, differentiable mechanism for dynamic information retrieval. Multi-head attention runs multiple attention patterns in parallel. The cost is O(n^2) computation, which is the defining constraint of transformer architectures.
