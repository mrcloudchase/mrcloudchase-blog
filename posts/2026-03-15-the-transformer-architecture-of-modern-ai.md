---
title: "The Transformer: Architecture of Modern AI"
date: "2026-03-15"
excerpt: "How attention, normalization, and residual connections combine into the architecture powering every modern LLM, from the original encoder-decoder to decoder-only GPT."
author: "Chase Dovey"
tags: ["AI", "Deep Learning"]
draft: true
---

## Introduction

Attention solved the information access problem. Every position can directly see every other position. No bottleneck, no vanishing gradients through time. But attention alone is not an architecture. It is a mechanism, one component in a larger system.

The transformer is that system. It takes the attention mechanism and wraps it in a specific arrangement of residual connections, layer normalization, and feed-forward networks that, together, create a stack of interchangeable processing blocks. Each block refines the representation. Stack enough blocks and the network learns to perform sophisticated reasoning, translation, code generation, and more.

This post covers the full architecture: every component, why it exists, and how the pieces fit together. We start with the transformer block itself, then cover the original encoder-decoder design from "Attention Is All You Need," and finish with the decoder-only architecture that powers modern LLMs like GPT, Claude, and LLaMA.

## The Transformer Block

Every transformer, encoder or decoder, is built from the same repeating unit: the **transformer block**. A single block contains four operations:

1. **Layer normalization**
2. **Multi-head attention**
3. **Residual connection** (skip connection)
4. **Feed-forward network** (another residual connection wraps this too)

The block takes a sequence of vectors in and produces a sequence of vectors out, with the same shape. This means blocks are stackable. The output of one block feeds directly into the next.

<svg viewBox="0 0 320 380" xmlns="http://www.w3.org/2000/svg" style="max-width:360px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="320" height="380" rx="8" fill="#181818"/>
  <text x="160" y="20" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">transformer block (pre-norm)</text>
  <!-- input arrow from bottom -->
  <line x1="160" y1="365" x2="160" y2="340" stroke="#4ade80" stroke-width="1.5"/>
  <text x="160" y="375" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">input</text>
  <!-- split for residual 1 -->
  <circle cx="160" cy="335" r="3" fill="#555"/>
  <line x1="160" y1="335" x2="60" y2="335" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="60" y1="335" x2="60" y2="225" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- layer norm 1 -->
  <rect x="110" y="300" width="100" height="28" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="160" y="318" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">LayerNorm</text>
  <line x1="160" y1="335" x2="160" y2="328" stroke="#4ade80" stroke-width="1.5"/>
  <!-- attention -->
  <line x1="160" y1="300" x2="160" y2="280" stroke="#888" stroke-width="1.5"/>
  <rect x="100" y="252" width="120" height="28" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="160" y="270" fill="#22d3ee" font-family="monospace" font-size="9" text-anchor="middle">Multi-Head Attn</text>
  <!-- add residual 1 -->
  <line x1="160" y1="252" x2="160" y2="232" stroke="#888" stroke-width="1.5"/>
  <circle cx="160" cy="225" r="8" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="160" y="229" fill="#fbbf24" font-family="monospace" font-size="11" text-anchor="middle">+</text>
  <line x1="60" y1="225" x2="152" y2="225" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- split for residual 2 -->
  <line x1="160" y1="217" x2="160" y2="210" stroke="#888" stroke-width="1"/>
  <circle cx="160" cy="205" r="3" fill="#555"/>
  <line x1="160" y1="205" x2="60" y2="205" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="60" y1="205" x2="60" y2="95" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- layer norm 2 -->
  <rect x="110" y="170" width="100" height="28" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="160" y="188" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">LayerNorm</text>
  <line x1="160" y1="205" x2="160" y2="198" stroke="#888" stroke-width="1"/>
  <!-- FFN -->
  <line x1="160" y1="170" x2="160" y2="150" stroke="#888" stroke-width="1.5"/>
  <rect x="100" y="122" width="120" height="28" rx="4" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="160" y="140" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">Feed-Forward</text>
  <!-- add residual 2 -->
  <line x1="160" y1="122" x2="160" y2="102" stroke="#888" stroke-width="1.5"/>
  <circle cx="160" cy="95" r="8" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="160" y="99" fill="#fbbf24" font-family="monospace" font-size="11" text-anchor="middle">+</text>
  <line x1="60" y1="95" x2="152" y2="95" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- output -->
  <line x1="160" y1="87" x2="160" y2="45" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="160" y="40" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">output</text>
  <!-- labels -->
  <text x="45" y="280" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">residual</text>
  <text x="45" y="150" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">residual</text>
  <!-- animated signal flowing through block -->
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M160,360 L160,335 L160,314 L160,266 L160,225 L160,205 L160,184 L160,136 L160,95 L160,50" dur="5s" repeatCount="indefinite" keyTimes="0;0.05;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.83;0.85;1" dur="5s" repeatCount="indefinite"/>
  </circle>
</svg>

The diagram shows the **pre-norm** variant, where layer normalization comes before each sub-layer. The original transformer paper used **post-norm** (normalize after the residual addition), but pre-norm is now standard because it produces more stable gradients during training and allows deeper stacking.

## Residual Connections

The residual connections (the dashed lines and `+` nodes in the diagram above) are arguably the most important structural feature of the transformer. Without them, stacking many layers causes gradient degradation, where the signal becomes too weak for early layers to learn effectively.

A residual connection adds the input of a sub-layer to its output:

```
output = x + SubLayer(x)
```

This means the sub-layer only needs to learn the **difference** between what it receives and what it should produce. Learning a small correction is easier than learning the entire transformation from scratch.

### The Residual Stream

There is a powerful way to think about transformers that comes from the residual connection structure. The input enters a **residual stream**, a running representation that flows through the entire network. Each attention layer and each feed-forward layer **reads from** the residual stream and **writes back to** it.

<svg viewBox="0 0 460 170" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="170" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">the residual stream</text>
  <!-- main stream line -->
  <line x1="30" y1="85" x2="440" y2="85" stroke="#fbbf24" stroke-width="2"/>
  <text x="20" y="89" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="end">in</text>
  <text x="448" y="89" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="start">out</text>
  <!-- layer 1 attention -->
  <rect x="60" y="35" width="50" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="85" y="50" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">attn 1</text>
  <line x1="75" y1="85" x2="75" y2="57" stroke="#22d3ee" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="95" y1="57" x2="95" y2="85" stroke="#22d3ee" stroke-width="1"/>
  <!-- layer 1 FFN -->
  <rect x="60" y="110" width="50" height="22" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="85" y="125" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">FFN 1</text>
  <line x1="75" y1="85" x2="75" y2="110" stroke="#4ade80" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="95" y1="110" x2="95" y2="85" stroke="#4ade80" stroke-width="1"/>
  <!-- layer 2 attention -->
  <rect x="160" y="35" width="50" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="185" y="50" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">attn 2</text>
  <line x1="175" y1="85" x2="175" y2="57" stroke="#22d3ee" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="195" y1="57" x2="195" y2="85" stroke="#22d3ee" stroke-width="1"/>
  <!-- layer 2 FFN -->
  <rect x="160" y="110" width="50" height="22" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="185" y="125" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">FFN 2</text>
  <line x1="175" y1="85" x2="175" y2="110" stroke="#4ade80" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="195" y1="110" x2="195" y2="85" stroke="#4ade80" stroke-width="1"/>
  <!-- dots -->
  <text x="245" y="89" fill="#555" font-family="monospace" font-size="12" text-anchor="middle">...</text>
  <!-- layer N attention -->
  <rect x="280" y="35" width="50" height="22" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="305" y="50" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">attn N</text>
  <line x1="295" y1="85" x2="295" y2="57" stroke="#22d3ee" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="315" y1="57" x2="315" y2="85" stroke="#22d3ee" stroke-width="1"/>
  <!-- layer N FFN -->
  <rect x="280" y="110" width="50" height="22" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="305" y="125" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">FFN N</text>
  <line x1="295" y1="85" x2="295" y2="110" stroke="#4ade80" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="315" y1="110" x2="315" y2="85" stroke="#4ade80" stroke-width="1"/>
  <!-- final LN -->
  <rect x="365" y="74" width="55" height="22" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="393" y="89" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">LayerNorm</text>
  <!-- legend -->
  <text x="85" y="155" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">dashed = read</text>
  <text x="200" y="155" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">solid = write back</text>
  <!-- animated flow along stream -->
  <circle r="3" fill="#fbbf24" opacity="0">
    <animateMotion path="M30,85 L95,85 L195,85 L315,85 L393,85 L440,85" dur="4s" repeatCount="indefinite" keyTimes="0;0.05;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.83;0.85;1" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>

This "residual stream" view explains several transformer properties:

- **Early layers can communicate with late layers** through the stream, without passing through intermediate attention or FFN computations
- **Layer deletion** experiments show you can remove some middle layers with minimal performance loss, because the residual stream preserves the original representation
- **Superposition**: The residual stream can carry more features than its dimensionality would suggest, because different components write to and read from different subspaces

## Layer Normalization

Layer normalization standardizes the activations across the feature dimension for each position independently:

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
```

Where `gamma` and `beta` are learned parameters (scale and shift), and `epsilon` is a small constant for numerical stability (typically 1e-5).

Why is this necessary? Without normalization, the scale of activations can drift across layers. One layer might produce values in the range [0, 1], the next in [0, 100]. This makes training unstable because the optimal learning rate depends on the activation scale. Layer normalization keeps the scale consistent.

### Pre-Norm vs Post-Norm

The original transformer paper placed layer norm **after** each sub-layer (post-norm):

```
output = LayerNorm(x + SubLayer(x))
```

Modern transformers use **pre-norm**, placing it before:

```
output = x + SubLayer(LayerNorm(x))
```

Pre-norm produces more stable gradients because the residual path is unobstructed. In post-norm, the gradient must flow through the normalization layer, which can amplify or suppress gradient components. With pre-norm, the gradient flows directly through the addition, and the sub-layer's contribution is normalized before it enters the residual stream.

The practical impact: pre-norm transformers train more stably at large depths and can often skip learning rate warmup. Post-norm transformers sometimes produce slightly better final performance but require careful tuning.

Some modern architectures use **RMSNorm** (Root Mean Square normalization) instead of full layer normalization:

```
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon)
```

RMSNorm drops the mean subtraction and the beta parameter, reducing computation. It works because the mean-centering is often redundant given the learned scale parameter.

## Positional Encoding

Self-attention is **permutation equivariant**: if you shuffle the input positions, you get the same output shuffled the same way. Attention has no inherent notion of position. Without positional encoding, the sentence "dog bites man" and "man bites dog" would produce identical attention patterns.

### Sinusoidal Positional Encoding (Original)

The original transformer added sinusoidal functions of different frequencies to the input embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension uses a different frequency. Low-frequency dimensions encode coarse position (beginning vs. end of sequence), while high-frequency dimensions encode fine position (adjacent tokens).

This encoding has a useful property: the encoding of position `pos + k` can be expressed as a linear transformation of the encoding at position `pos`, for any fixed offset `k`. This means the model can learn to attend to relative positions using linear operations.

<svg viewBox="0 0 460 180" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="180" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">sinusoidal positional encoding</text>
  <!-- axes -->
  <line x1="50" y1="140" x2="430" y2="140" stroke="#555" stroke-width="1"/>
  <line x1="50" y1="35" x2="50" y2="140" stroke="#555" stroke-width="1"/>
  <text x="240" y="160" fill="#888" font-family="monospace" font-size="8" text-anchor="middle">position in sequence</text>
  <text x="25" y="90" fill="#888" font-family="monospace" font-size="8" text-anchor="middle" transform="rotate(-90,25,90)">value</text>
  <!-- high frequency sine (dim 0) -->
  <path d="M50,87 Q60,45 70,87 Q80,130 90,87 Q100,45 110,87 Q120,130 130,87 Q140,45 150,87 Q160,130 170,87 Q180,45 190,87 Q200,130 210,87 Q220,45 230,87 Q240,130 250,87 Q260,45 270,87 Q280,130 290,87 Q300,45 310,87 Q320,130 330,87 Q340,45 350,87 Q360,130 370,87 Q380,45 390,87 Q400,130 410,87 Q420,45 430,87" fill="none" stroke="#22d3ee" stroke-width="1.2" opacity="0.8"/>
  <!-- medium frequency sine (dim 4) -->
  <path d="M50,87 Q90,35 130,87 Q170,140 210,87 Q250,35 290,87 Q330,140 370,87 Q410,35 430,75" fill="none" stroke="#4ade80" stroke-width="1.2" opacity="0.8"/>
  <!-- low frequency sine (dim 8) -->
  <path d="M50,87 Q140,30 230,87 Q320,145 410,87 L430,82" fill="none" stroke="#fbbf24" stroke-width="1.2" opacity="0.8"/>
  <!-- legend -->
  <line x1="60" y1="170" x2="75" y2="170" stroke="#22d3ee" stroke-width="1.2"/>
  <text x="80" y="173" fill="#22d3ee" font-family="monospace" font-size="7">dim 0 (high freq)</text>
  <line x1="190" y1="170" x2="205" y2="170" stroke="#4ade80" stroke-width="1.2"/>
  <text x="210" y="173" fill="#4ade80" font-family="monospace" font-size="7">dim 4 (med freq)</text>
  <line x1="330" y1="170" x2="345" y2="170" stroke="#fbbf24" stroke-width="1.2"/>
  <text x="350" y="173" fill="#fbbf24" font-family="monospace" font-size="7">dim 8 (low freq)</text>
  <!-- position markers -->
  <text x="50" y="150" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">0</text>
  <text x="145" y="150" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">25</text>
  <text x="240" y="150" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">50</text>
  <text x="335" y="150" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">75</text>
  <text x="430" y="150" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">100</text>
</svg>

### Learned Positional Embeddings

An alternative: learn a separate embedding vector for each position, just like token embeddings. Position 0 gets a learned vector, position 1 gets another, and so on. Simple and effective for fixed-length contexts.

The limitation is that learned embeddings cannot generalize beyond the maximum position seen during training. If you train with 2048 positions, position 2049 has no embedding.

### Rotary Position Embedding (RoPE)

Modern LLMs predominantly use **Rotary Position Embedding** (RoPE), which encodes position by rotating the query and key vectors in pairs of dimensions:

```
RoPE(x, pos) = x * cos(pos * theta) + rotate(x) * sin(pos * theta)
```

Where `theta` varies by dimension pair, creating rotations at different frequencies (similar to sinusoidal encoding). The key insight: the dot product between two RoPE-encoded vectors depends only on their relative position, not their absolute positions.

RoPE advantages:
- **Relative position awareness**: Attention scores naturally capture distance between tokens
- **Extrapolation**: With techniques like NTK-aware scaling or YaRN, RoPE can extend to sequence lengths beyond training
- **No additional parameters**: The rotation is computed from position, not learned
- **Decays with distance**: The expected dot product between distant positions naturally decreases, providing a useful inductive bias

## The Feed-Forward Network

Each transformer block contains a position-wise feed-forward network (FFN) that processes each position independently (no cross-position interaction, that is attention's job). The original design:

```
FFN(x) = activation(x . W_1 + b_1) . W_2 + b_2
```

The critical detail: the inner dimension `d_ff` is larger than `d_model`. Typically `d_ff = 4 * d_model`. For a model with `d_model = 4096`, the FFN expands to 16384 dimensions and then contracts back to 4096.

Why expand and contract? The expansion creates a high-dimensional space where the network can represent complex nonlinear functions. The contraction projects the result back to the model dimension. This is where the transformer stores **factual knowledge**. Research has shown that individual neurons in the FFN activate for specific concepts, patterns, or facts.

### Gated FFN Variants

Modern transformers use gated variants of the FFN, most commonly **SwiGLU**:

```
SwiGLU(x) = (x . W_1 * SiLU(x . W_gate)) . W_2
```

Where `SiLU(x) = x * sigmoid(x)`. The gating mechanism (`x . W_gate`) provides the network with a multiplicative interaction that improves expressiveness. The cost: three weight matrices instead of two, so `d_ff` is typically reduced to `8/3 * d_model` to keep the parameter count comparable.

## The Encoder-Decoder Transformer

The original "Attention Is All You Need" (Vaswani et al., 2017) introduced a transformer with two stacks: an **encoder** and a **decoder**, connected by **cross-attention**.

### The Encoder Stack

The encoder processes the input sequence with **bidirectional** self-attention. Every position can attend to every other position, forward and backward. This is ideal for tasks where you have the entire input available at once (translation, classification, summarization of a given text).

Each encoder layer is a standard transformer block: layer norm, multi-head self-attention, residual connection, layer norm, FFN, residual connection.

### The Decoder Stack

The decoder generates the output sequence one token at a time. It has two attention mechanisms per layer:

1. **Masked self-attention**: The decoder can only attend to previous positions in the output sequence. Position 5 can see positions 0-4 but not positions 6 onward. This prevents the model from "cheating" by looking at future tokens during training.

2. **Cross-attention**: After self-attention, the decoder attends to the encoder's output. Queries come from the decoder, but keys and values come from the encoder. This is how the decoder accesses the input sequence.

<svg viewBox="0 0 460 320" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="320" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">encoder-decoder transformer</text>
  <!-- ENCODER SIDE -->
  <text x="120" y="40" text-anchor="middle" fill="#4ade80" font-family="monospace" font-size="9">encoder</text>
  <!-- encoder block -->
  <rect x="55" y="48" width="130" height="115" rx="6" fill="#222" stroke="#4ade80" stroke-width="1" stroke-dasharray="4,2"/>
  <rect x="70" y="130" width="100" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="120" y="145" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">LayerNorm</text>
  <rect x="70" y="100" width="100" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="120" y="115" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">FFN</text>
  <rect x="70" y="70" width="100" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="120" y="85" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">LayerNorm</text>
  <rect x="70" y="52" width="100" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="120" y="67" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Self-Attention</text>
  <text x="120" y="175" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">x N layers</text>
  <!-- encoder input -->
  <line x1="120" y1="210" x2="120" y2="163" stroke="#4ade80" stroke-width="1"/>
  <rect x="60" y="210" width="120" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="120" y="225" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">Input Embedding</text>
  <rect x="60" y="240" width="120" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="120" y="255" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">+ Pos Encoding</text>
  <text x="120" y="280" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">source tokens</text>
  <!-- DECODER SIDE -->
  <text x="340" y="40" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9">decoder</text>
  <!-- decoder block -->
  <rect x="270" y="48" width="140" height="145" rx="6" fill="#222" stroke="#22d3ee" stroke-width="1" stroke-dasharray="4,2"/>
  <rect x="285" y="160" width="110" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="340" y="175" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">LayerNorm</text>
  <rect x="285" y="130" width="110" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="340" y="145" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">FFN</text>
  <rect x="285" y="100" width="110" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="340" y="115" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">LayerNorm</text>
  <rect x="285" y="75" width="110" height="22" rx="3" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="340" y="90" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">Cross-Attention</text>
  <rect x="285" y="52" width="110" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="340" y="67" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Masked Self-Attn</text>
  <text x="340" y="205" fill="#555" font-family="monospace" font-size="7" text-anchor="middle">x N layers</text>
  <!-- decoder input -->
  <line x1="340" y1="240" x2="340" y2="193" stroke="#22d3ee" stroke-width="1"/>
  <rect x="280" y="240" width="120" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="340" y="255" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Output Embedding</text>
  <rect x="280" y="270" width="120" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="340" y="285" fill="#888" font-family="monospace" font-size="7" text-anchor="middle">+ Pos Encoding</text>
  <text x="340" y="310" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">target tokens (shifted)</text>
  <!-- cross-attention bridge -->
  <line x1="185" y1="63" x2="285" y2="86" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="230" y="68" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">K, V</text>
  <!-- output -->
  <line x1="340" y1="48" x2="340" y2="30" stroke="#fbbf24" stroke-width="1"/>
  <text x="400" y="35" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="start">output probs</text>
  <!-- animated cross-attention signal -->
  <circle r="3" fill="#fbbf24" opacity="0">
    <animateMotion path="M120,63 L185,63 L235,75 L285,86 L340,86" dur="3s" repeatCount="indefinite" keyTimes="0;0.05;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.83;0.85;1" dur="3s" repeatCount="indefinite"/>
  </circle>
</svg>

Cross-attention is the bridge between encoder and decoder. The encoder processes the full input once, producing a set of key-value pairs. The decoder then queries these key-value pairs at every layer and every generation step. This means the encoder's computation is amortized: it runs once, but the decoder references its output repeatedly.

## The Decoder-Only Transformer

The encoder-decoder design works well for sequence-to-sequence tasks where the input and output are distinct (translation, summarization). But for language modeling, where the task is simply to predict the next token, the encoder is unnecessary. The decoder alone suffices.

This is the **decoder-only** transformer, the architecture behind GPT, Claude, LLaMA, and most modern LLMs. Remove the encoder. Remove cross-attention. Keep only the decoder stack with its masked self-attention and feed-forward layers.

The input and output use the same vocabulary, the same embedding space, and the same sequence. The model reads a sequence of tokens and predicts the next token at each position.

### Causal Masking

The defining feature of the decoder-only transformer is the **causal attention mask**. When predicting the token at position `t`, the model can only attend to positions `0, 1, ..., t`. It cannot see the future.

This is enforced by adding a mask to the attention scores before softmax:

```
scores = Q . K^T / sqrt(d_k)
scores = scores + mask    # mask is 0 for allowed, -inf for blocked
attention = softmax(scores) . V
```

The mask is a lower-triangular matrix: positions above the diagonal are set to negative infinity, which softmax converts to zero probability.

<svg viewBox="0 0 380 200" xmlns="http://www.w3.org/2000/svg" style="max-width:420px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="380" height="200" rx="8" fill="#181818"/>
  <text x="190" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">causal attention mask</text>
  <!-- matrix grid -->
  <text x="85" y="40" fill="#888" font-family="monospace" font-size="8" text-anchor="middle">keys (attending to)</text>
  <text x="30" y="110" fill="#888" font-family="monospace" font-size="8" text-anchor="middle" transform="rotate(-90,30,110)">queries (from)</text>
  <!-- column labels -->
  <text x="72" y="55" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">t0</text>
  <text x="97" y="55" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">t1</text>
  <text x="122" y="55" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">t2</text>
  <text x="147" y="55" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">t3</text>
  <text x="172" y="55" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">t4</text>
  <!-- row labels -->
  <text x="52" y="75" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="end">t0</text>
  <text x="52" y="100" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="end">t1</text>
  <text x="52" y="125" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="end">t2</text>
  <text x="52" y="150" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="end">t3</text>
  <text x="52" y="175" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="end">t4</text>
  <!-- row 0: allowed -->
  <rect x="60" y="62" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="72" y="76" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="85" y="62" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="97" y="76" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <rect x="110" y="62" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="122" y="76" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <rect x="135" y="62" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="147" y="76" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <rect x="160" y="62" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="172" y="76" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <!-- row 1 -->
  <rect x="60" y="87" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="72" y="101" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="85" y="87" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="97" y="101" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="110" y="87" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="122" y="101" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <rect x="135" y="87" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="147" y="101" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <rect x="160" y="87" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="172" y="101" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <!-- row 2 -->
  <rect x="60" y="112" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="72" y="126" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="85" y="112" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="97" y="126" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="110" y="112" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="122" y="126" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="135" y="112" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="147" y="126" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <rect x="160" y="112" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="172" y="126" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <!-- row 3 -->
  <rect x="60" y="137" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="72" y="151" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="85" y="137" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="97" y="151" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="110" y="137" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="122" y="151" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="135" y="137" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="147" y="151" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="160" y="137" width="25" height="20" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="172" y="151" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="middle">-inf</text>
  <!-- row 4 -->
  <rect x="60" y="162" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="72" y="176" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="85" y="162" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="97" y="176" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="110" y="162" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="122" y="176" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="135" y="162" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="147" y="176" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <rect x="160" y="162" width="25" height="20" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="172" y="176" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">1</text>
  <!-- explanation -->
  <rect x="210" y="65" width="12" height="12" rx="2" fill="#4ade80" opacity="0.3"/>
  <text x="228" y="75" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="start">= can attend</text>
  <rect x="210" y="85" width="12" height="12" rx="2" fill="#ef4444" opacity="0.15"/>
  <text x="228" y="95" fill="#ef4444" font-family="monospace" font-size="7" text-anchor="start">= masked (future)</text>
  <text x="285" y="125" fill="#888" font-family="monospace" font-size="7" text-anchor="start">token t2 can see</text>
  <text x="285" y="135" fill="#888" font-family="monospace" font-size="7" text-anchor="start">t0, t1, t2 only</text>
  <text x="285" y="155" fill="#888" font-family="monospace" font-size="7" text-anchor="start">lower triangular</text>
  <text x="285" y="165" fill="#888" font-family="monospace" font-size="7" text-anchor="start">= causal mask</text>
</svg>

During training, causal masking allows the model to compute the loss for **every position simultaneously**. Token at position 0 predicts position 1, position 1 predicts position 2, and so on. This is much more efficient than generating one token at a time, because a single forward pass produces N-1 training signals from a sequence of N tokens.

### The Decoder-Only Architecture

The full decoder-only transformer:

<svg viewBox="0 0 340 360" xmlns="http://www.w3.org/2000/svg" style="max-width:380px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="340" height="360" rx="8" fill="#181818"/>
  <text x="170" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">decoder-only transformer</text>
  <!-- input tokens -->
  <text x="170" y="348" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">input tokens</text>
  <line x1="170" y1="338" x2="170" y2="320" stroke="#4ade80" stroke-width="1"/>
  <!-- token embedding -->
  <rect x="100" y="298" width="140" height="22" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="170" y="313" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">Token Embedding</text>
  <!-- positional encoding -->
  <line x1="170" y1="298" x2="170" y2="285" stroke="#888" stroke-width="1"/>
  <rect x="100" y="265" width="140" height="22" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="170" y="280" fill="#888" font-family="monospace" font-size="8" text-anchor="middle">+ RoPE / Pos Encoding</text>
  <!-- transformer blocks -->
  <line x1="170" y1="265" x2="170" y2="250" stroke="#888" stroke-width="1"/>
  <rect x="65" y="100" width="210" height="150" rx="6" fill="#222" stroke="#555" stroke-width="1" stroke-dasharray="4,2"/>
  <text x="280" y="180" fill="#555" font-family="monospace" font-size="8" text-anchor="start">x N</text>
  <!-- block contents -->
  <rect x="95" y="215" width="150" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="170" y="230" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">RMSNorm</text>
  <rect x="95" y="185" width="150" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="170" y="200" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Causal Multi-Head Attn</text>
  <rect x="95" y="155" width="150" height="22" rx="3" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="170" y="170" fill="#ccc" font-family="monospace" font-size="7" text-anchor="middle">RMSNorm</text>
  <rect x="95" y="125" width="150" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="170" y="140" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">SwiGLU FFN</text>
  <rect x="95" y="105" width="150" height="16" rx="3" fill="none" stroke="#fbbf24" stroke-width="1" stroke-dasharray="2,2"/>
  <text x="170" y="116" fill="#fbbf24" font-family="monospace" font-size="6" text-anchor="middle">+ residual connections</text>
  <!-- final layer norm -->
  <line x1="170" y1="100" x2="170" y2="85" stroke="#888" stroke-width="1"/>
  <rect x="100" y="65" width="140" height="22" rx="4" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="170" y="80" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">Final RMSNorm</text>
  <!-- output head -->
  <line x1="170" y1="65" x2="170" y2="50" stroke="#888" stroke-width="1"/>
  <rect x="100" y="30" width="140" height="22" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="170" y="45" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">Linear + Softmax</text>
  <!-- output -->
  <text x="260" y="45" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="start">next token</text>
  <text x="260" y="55" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="start">probabilities</text>
  <!-- animated signal -->
  <circle r="3" fill="#22d3ee" opacity="0">
    <animateMotion path="M170,338 L170,306 L170,276 L170,230 L170,196 L170,166 L170,136 L170,76 L170,41" dur="5s" repeatCount="indefinite" keyTimes="0;0.05;0.85;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.05;0.83;0.85;1" dur="5s" repeatCount="indefinite"/>
  </circle>
</svg>

The simplicity is the point. One block type, repeated N times. No encoder, no cross-attention. The same architecture handles any text task: translation (include the source text in the prompt), summarization (include the document in the prompt), question answering (include the context in the prompt). The task is specified through the input, not the architecture.

## KV Cache: Making Inference Efficient

During autoregressive generation, the model generates one token at a time. Each new token requires a full forward pass. But there is a massive redundancy: when generating token `t+1`, the keys and values for positions `0` through `t` are exactly the same as they were when generating token `t`.

The **KV cache** stores these previously computed keys and values, so they do not need to be recomputed:

```
# Without KV cache (wasteful):
# For each new token, recompute K and V for ALL positions

# With KV cache (efficient):
# Step 1: Compute K, V for token 0. Store in cache.
# Step 2: Compute K, V for token 1 only. Append to cache.
#          Attention uses full cache [K0, K1] and [V0, V1].
# Step 3: Compute K, V for token 2 only. Append to cache.
#          Attention uses full cache [K0, K1, K2] and [V0, V1, V2].
# ...
```

Without the KV cache, generating a sequence of length `n` requires O(n^2) total computation (each step recomputes attention over all previous positions). With the KV cache, each step computes attention for just the new token against the cached keys and values: O(n) per step, O(n^2) total, but with a much smaller constant factor.

<svg viewBox="0 0 460 190" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="460" height="190" rx="8" fill="#181818"/>
  <text x="230" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">KV cache during generation</text>
  <!-- step labels -->
  <text x="30" y="52" fill="#888" font-family="monospace" font-size="8" text-anchor="end">step 1</text>
  <text x="30" y="92" fill="#888" font-family="monospace" font-size="8" text-anchor="end">step 2</text>
  <text x="30" y="132" fill="#888" font-family="monospace" font-size="8" text-anchor="end">step 3</text>
  <text x="30" y="172" fill="#888" font-family="monospace" font-size="8" text-anchor="end">step 4</text>
  <!-- step 1 -->
  <rect x="40" y="38" width="55" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="68" y="53" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Q: "The"</text>
  <rect x="110" y="38" width="55" height="22" rx="3" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="138" y="53" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">K: "The"</text>
  <rect x="180" y="38" width="55" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="208" y="53" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">V: "The"</text>
  <text x="260" y="53" fill="#999" font-family="monospace" font-size="7" text-anchor="start">-> "cat"</text>
  <!-- step 2 -->
  <rect x="40" y="78" width="55" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="68" y="93" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Q: "cat"</text>
  <rect x="110" y="78" width="112" height="22" rx="3" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="166" y="93" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">cached K + new K</text>
  <rect x="237" y="78" width="112" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="293" y="93" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">cached V + new V</text>
  <text x="370" y="93" fill="#999" font-family="monospace" font-size="7" text-anchor="start">-> "sat"</text>
  <!-- step 3 -->
  <rect x="40" y="118" width="55" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="68" y="133" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Q: "sat"</text>
  <rect x="110" y="118" width="145" height="22" rx="3" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="183" y="133" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">cached K (growing)</text>
  <rect x="270" y="118" width="145" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="343" y="133" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">cached V (growing)</text>
  <text x="430" y="133" fill="#999" font-family="monospace" font-size="7" text-anchor="start">-></text>
  <!-- step 4 -->
  <rect x="40" y="158" width="55" height="22" rx="3" fill="#333" stroke="#22d3ee" stroke-width="1"/>
  <text x="68" y="173" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">Q: new</text>
  <rect x="110" y="158" width="180" height="22" rx="3" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="200" y="173" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">cached K (still growing)</text>
  <rect x="305" y="158" width="140" height="22" rx="3" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="375" y="173" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">cached V (growing)</text>
  <!-- annotation -->
  <text x="68" y="35" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">only new token</text>
  <text x="200" y="35" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">reuse previous K, V</text>
</svg>

The trade-off: the KV cache consumes memory proportional to `batch_size * num_layers * 2 * num_heads * d_head * seq_len`. For a 70B parameter model with a 4096-token context, the KV cache alone can consume tens of gigabytes of GPU memory.

## Grouped-Query Attention (GQA)

The KV cache memory problem led to **Grouped-Query Attention** (GQA), a middle ground between standard multi-head attention (MHA) and multi-query attention (MQA).

In standard MHA, each attention head has its own Q, K, and V projections. With 32 heads, you store 32 sets of keys and values in the cache.

In MQA, all heads share a single K and V. This reduces cache size by a factor of `num_heads`, but can hurt quality because the heads lose the ability to attend to different information.

GQA groups the heads. With 32 query heads and 8 KV groups, every 4 query heads share the same K and V:

```
Standard MHA: 32 Q heads, 32 K heads, 32 V heads  (32x cache)
GQA:          32 Q heads,  8 K heads,  8 V heads   ( 8x cache)
MQA:          32 Q heads,  1 K head,   1 V head    ( 1x cache)
```

GQA reduces KV cache memory by 4x (in this example) with minimal quality loss. LLaMA 2 70B, Mistral, and many modern models use GQA.

## The Training Objective: Next-Token Prediction

The decoder-only transformer is trained with a remarkably simple objective: given all previous tokens, predict the next one.

```
Loss = -sum over t: log P(token_t | token_0, ..., token_{t-1})
```

This is the cross-entropy loss between the model's predicted probability distribution over the vocabulary and the actual next token. The model produces a probability for every token in the vocabulary (typically 32K-128K tokens) at every position, and the loss penalizes assigning low probability to the correct token.

The elegance of this objective: it requires no labeled data. Any text is training data. The internet contains trillions of tokens. To predict the next word well, the model must learn grammar, facts, reasoning, coding, mathematics, and more. Next-token prediction is a universal task that forces the emergence of general intelligence.

### Teacher Forcing

During training, the model sees the actual tokens at every position (not its own predictions). This is called **teacher forcing**. At position `t`, the input is always the ground truth token, regardless of what the model would have predicted. This provides stable gradients and allows the parallel computation of loss at all positions simultaneously.

During inference, there is no ground truth. The model generates token by token, feeding its own predictions back as input. This mismatch between training (teacher forcing) and inference (autoregressive) is called **exposure bias**, but in practice, well-trained large models handle it with negligible degradation.

## Scaling Laws

One of the most consequential discoveries in modern AI is that transformer performance follows predictable **scaling laws**. Loss decreases as a power law with respect to three factors:

```
L(N, D, C) ~ a/N^alpha + b/D^beta + c
```

Where `N` is the number of parameters, `D` is the amount of training data, and `C` is the compute budget (FLOPs).

### The Kaplan Scaling Laws (2020)

OpenAI's initial scaling laws paper found that loss scales smoothly with model size, dataset size, and compute. Larger models are more sample-efficient: they extract more from the same data.

### Chinchilla (2022)

DeepMind's Chinchilla paper refined the scaling laws and discovered that most large models were **undertrained**. The compute-optimal ratio is roughly 20 tokens per parameter. A 70B model should be trained on ~1.4 trillion tokens.

This had immediate impact. Instead of training a 280B model on 300B tokens (like Gopher), train a 70B model on 1.4T tokens. Same compute, better performance. Chinchilla outperformed models 4x its size.

### Modern Scaling

Current practice often "over-trains" relative to Chinchilla-optimal, training smaller models on much more data than the Chinchilla ratio suggests. The reason: inference cost scales with model size, not training data. A smaller model trained on more data is cheaper to run, even if the training itself costs more.

LLaMA 7B was trained on 1T tokens (143x Chinchilla ratio). LLaMA 2 7B on 2T tokens. The reasoning: training happens once, but inference happens billions of times.

## The Complete Lineage

The transformer did not appear from nowhere. Every component has a lineage:

**McCulloch-Pitts (1943)** gave us the artificial neuron: weighted inputs through a threshold function. **Rosenblatt (1958)** added learning: the perceptron updates its own weights. **Backpropagation (1986)** extended learning to deep networks: multi-layer perceptrons with many hidden layers. **Attention (2014)** solved the long-range dependency problem: dynamic, learnable focus. **The Transformer (2017)** assembled the pieces: attention, residual connections, layer normalization, and feed-forward networks into a stackable block that scales predictably with compute.

Each component solves a specific problem:

| Component | Problem it solves |
|---|---|
| Weighted sum | Combining multiple signals |
| Activation function | Introducing nonlinearity |
| Backpropagation | Learning in deep networks |
| Residual connections | Training very deep networks |
| Layer normalization | Stable training dynamics |
| Attention | Long-range dependencies |
| Positional encoding | Sequence order |
| Causal masking | Autoregressive generation |
| KV cache | Efficient inference |
| GQA | Memory-efficient attention |

The transformer is not a single invention. It is the careful composition of a dozen solutions, each addressing a specific failure mode of earlier architectures. Biological neurons inspired mathematical neurons. Mathematical neurons stacked into perceptrons. Perceptrons stacked into deep networks. Deep networks gained attention. Attention gained structure. And the result handles language, code, images, audio, video, and robotics with the same architecture.

That is the transformer. Not a breakthrough, but a synthesis.

## Key Takeaways

The transformer block is a residual stream with two sub-layers: multi-head attention and a feed-forward network, each preceded by layer normalization. Residual connections enable the training of very deep networks and create a continuous stream of information flow. Positional encoding injects sequence order into an otherwise position-agnostic architecture. The original encoder-decoder design uses cross-attention to bridge input and output sequences. The decoder-only simplification, with causal masking, powers every modern LLM. The KV cache makes autoregressive generation practical by avoiding redundant computation. GQA reduces cache memory with minimal quality loss. Next-token prediction, the training objective, is the universal task that drives the emergence of general capabilities. And scaling laws tell us that more compute, more data, and more parameters predictably improve performance, a finding that has shaped the entire trajectory of modern AI.
