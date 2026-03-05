---
title: "The Biological Neuron and Its Mathematical Model"
date: "2026-03-04"
excerpt: "From dendrites to equations. How researchers looked at a real brain cell and distilled it into the mathematical model that launched artificial intelligence."
author: "Chase Dovey"
tags: ["AI", "Deep Learning"]
draft: false
---

## Introduction

Every neural network including the current transformer based LLMs can trace their roots back to a single biological structure: the neuron. In 1943, Warren McCulloch and Walter Pitts looked at a neuron and asked themselves a question: what is the simplest mathematical model that captures what a neuron does?

Their answer threw away the complexities of biology and kept only the essence: weighted inputs, a sum, a threshold, a binary output. This distillation turned out to be one of the most consequential mathematical models in the history of computing.

This post covers both sides: the biology they started with and the mathematics they extracted from it. Understanding what they kept and what they discarded makes every subsequent development in deep learning clearer.

## The Neuron: Basic Anatomy

The human brain contains roughly 86 billion neurons. Each one is a specialized cell optimized for one job: receive signals, decide whether to fire, and transmit the result. Despite enormous variation in shape and size across the nervous system, every neuron shares the same four functional components.

<svg viewBox="0 0 660 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Biological neuron with animated signal mechanics: graded potentials decay along dendrites to the soma, summation triggers the axon hillock, the action potential jumps node-to-node via saltatory conduction, and neurotransmitter vesicles release at synaptic boutons." style="max-width:700px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Biological neuron anatomy and signal mechanics: dendritic spines, passive graded potentials, soma integration, hillock threshold, saltatory conduction at Nodes of Ranvier, and vesicle release at synaptic boutons</title>
  <rect width="660" height="220" rx="8" fill="#181818"/>
  <text x="330" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">neuron anatomy + signal mechanics</text>
  <!-- === DENDRITES WITH SPINES === -->
  <!-- top branch -->
  <path d="M62,42 Q92,52 115,67 Q133,78 155,88" fill="none" stroke="#4ade80" stroke-width="1.5"/>
  <path d="M62,42 Q48,32 35,24" fill="none" stroke="#4ade80" stroke-width="1"/>
  <path d="M62,42 Q52,54 42,62" fill="none" stroke="#4ade80" stroke-width="1"/>
  <circle cx="32" cy="22" r="3.5" fill="#4ade80" opacity="0.6"/>
  <circle cx="39" cy="64" r="3.5" fill="#4ade80" opacity="0.6"/>
  <!-- middle branch -->
  <path d="M57,102 Q87,100 115,97 Q137,94 155,94" fill="none" stroke="#4ade80" stroke-width="1.5"/>
  <path d="M57,102 Q43,90 32,80" fill="none" stroke="#4ade80" stroke-width="1"/>
  <path d="M57,102 Q43,114 32,124" fill="none" stroke="#4ade80" stroke-width="1"/>
  <circle cx="29" cy="78" r="3.5" fill="#4ade80" opacity="0.6"/>
  <circle cx="29" cy="126" r="3.5" fill="#4ade80" opacity="0.6"/>
  <!-- bottom branch -->
  <path d="M62,158 Q92,150 115,133 Q133,118 155,108" fill="none" stroke="#4ade80" stroke-width="1.5"/>
  <path d="M62,158 Q52,146 42,138" fill="none" stroke="#4ade80" stroke-width="1"/>
  <path d="M62,158 Q48,168 35,176" fill="none" stroke="#4ade80" stroke-width="1"/>
  <circle cx="39" cy="136" r="3.5" fill="#4ade80" opacity="0.6"/>
  <circle cx="32" cy="178" r="3.5" fill="#4ade80" opacity="0.6"/>
  <!-- === SOMA WITH NUCLEUS === -->
  <ellipse cx="195" cy="100" rx="40" ry="30" fill="#333" stroke="#4ade80" stroke-width="2">
    <animate attributeName="stroke-width" values="2;2;3;3.5;3.5;3;2;2" keyTimes="0;0.25;0.30;0.34;0.37;0.40;0.43;1" dur="10s" repeatCount="indefinite"/>
    <animate attributeName="stroke" values="#4ade80;#4ade80;#6aeda0;#86efac;#86efac;#6aeda0;#4ade80;#4ade80" keyTimes="0;0.25;0.30;0.34;0.37;0.40;0.43;1" dur="10s" repeatCount="indefinite"/>
  </ellipse>
  <circle cx="195" cy="97" r="13" fill="none" stroke="#555" stroke-width="1" stroke-dasharray="2"/>
  <text x="195" y="101" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">soma</text>
  <!-- === AXON HILLOCK (tapered funnel with Na+ channel density marks) === -->
  <path d="M235,88 L258,96 L258,104 L235,112 Z" fill="#222" stroke="#22d3ee" stroke-width="1.2"/>
  <line x1="242" y1="91" x2="242" y2="109" stroke="#22d3ee" stroke-width="0.5" opacity="0.3"/>
  <line x1="247" y1="93" x2="247" y2="107" stroke="#22d3ee" stroke-width="0.5" opacity="0.3"/>
  <line x1="252" y1="95" x2="252" y2="105" stroke="#22d3ee" stroke-width="0.5" opacity="0.3"/>
  <!-- === AXON === -->
  <line x1="258" y1="100" x2="480" y2="100" stroke="#22d3ee" stroke-width="2"/>
  <!-- === MYELIN SHEATH SEGMENTS (filled oblongs) === -->
  <rect x="266" y="89" width="34" height="22" rx="11" fill="#252525" stroke="#555" stroke-width="1"/>
  <rect x="316" y="89" width="34" height="22" rx="11" fill="#252525" stroke="#555" stroke-width="1"/>
  <rect x="366" y="89" width="34" height="22" rx="11" fill="#252525" stroke="#555" stroke-width="1"/>
  <rect x="416" y="89" width="34" height="22" rx="11" fill="#252525" stroke="#555" stroke-width="1"/>
  <!-- === AXON TERMINALS WITH BOUTONS === -->
  <!-- top terminal branch -->
  <path d="M480,100 Q498,78 512,62" fill="none" stroke="#fbbf24" stroke-width="1.5"/>
  <path d="M512,62 Q525,48 537,40" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <path d="M512,62 Q527,56 540,50" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <circle cx="540" cy="38" r="4" fill="#fbbf24" opacity="0.6"/>
  <circle cx="543" cy="50" r="4" fill="#fbbf24" opacity="0.6"/>
  <!-- middle terminal branch -->
  <path d="M480,100 Q505,100 522,98" fill="none" stroke="#fbbf24" stroke-width="1.5"/>
  <path d="M522,98 Q538,88 550,80" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <path d="M522,98 Q538,108 550,116" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <circle cx="553" cy="78" r="4" fill="#fbbf24" opacity="0.6"/>
  <circle cx="553" cy="118" r="4" fill="#fbbf24" opacity="0.6"/>
  <!-- bottom terminal branch -->
  <path d="M480,100 Q498,122 512,138" fill="none" stroke="#fbbf24" stroke-width="1.5"/>
  <path d="M512,138 Q525,150 537,158" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <path d="M512,138 Q527,146 540,150" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <circle cx="540" cy="160" r="4" fill="#fbbf24" opacity="0.6"/>
  <circle cx="543" cy="152" r="4" fill="#fbbf24" opacity="0.6"/>
  <!-- === SYNAPTIC CLEFTS (gap + postsynaptic membrane at select boutons) === -->
  <line x1="548" y1="34" x2="548" y2="42" stroke="#4ade80" stroke-width="1.5" opacity="0.4"/>
  <line x1="561" y1="74" x2="561" y2="82" stroke="#4ade80" stroke-width="1.5" opacity="0.4"/>
  <line x1="548" y1="156" x2="548" y2="164" stroke="#4ade80" stroke-width="1.5" opacity="0.4"/>
  <!-- === LABELS === -->
  <text x="18" y="100" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">spines</text>
  <text x="62" y="205" fill="#4ade80" font-family="monospace" font-size="9" text-anchor="middle">dendrites</text>
  <text x="247" y="126" fill="#22d3ee" font-family="monospace" font-size="8" text-anchor="middle">hillock</text>
  <text x="350" y="80" fill="#999" font-family="monospace" font-size="9" text-anchor="middle">myelin</text>
  <text x="305" y="126" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">node</text>
  <text x="355" y="126" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">node</text>
  <text x="405" y="126" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">node</text>
  <text x="550" y="205" fill="#fbbf24" font-family="monospace" font-size="9" text-anchor="middle">boutons</text>
  <!-- ============================================================ -->
  <!-- ANIMATION: 10s cycle with 8 mechanistic phases               -->
  <!-- ============================================================ -->
  <!-- PHASE 1: Graded potentials travel dendrites with DECAY (0-3s) -->
  <!-- top spine 1 (32,22) -> junction (62,42) -> soma edge (155,88) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M32,22 Q48,32 62,42 Q92,52 115,67 Q133,78 155,88" dur="10s" repeatCount="indefinite" keyTimes="0;0.02;0.28;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.8;0.5;0.25;0;0" keyTimes="0;0.02;0.14;0.24;0.28;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- top spine 2 (39,64) -> junction (62,42) -> soma edge (155,88) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M39,64 Q52,54 62,42 Q92,52 115,67 Q133,78 155,88" dur="10s" repeatCount="indefinite" keyTimes="0;0.04;0.28;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.8;0.5;0.25;0;0" keyTimes="0;0.04;0.14;0.24;0.28;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- middle spine 1 (29,78) -> junction (57,102) -> soma edge (155,94) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M29,78 Q43,90 57,102 Q87,100 115,97 Q137,94 155,94" dur="10s" repeatCount="indefinite" keyTimes="0;0.05;0.28;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.8;0.5;0.25;0;0" keyTimes="0;0.05;0.15;0.24;0.28;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- middle spine 2 (29,126) -> junction (57,102) -> soma edge (155,94) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M29,126 Q43,114 57,102 Q87,100 115,97 Q137,94 155,94" dur="10s" repeatCount="indefinite" keyTimes="0;0.07;0.28;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.8;0.5;0.25;0;0" keyTimes="0;0.07;0.16;0.24;0.28;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- bottom spine 1 (39,136) -> junction (62,158) -> soma edge (155,108) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M39,136 Q52,146 62,158 Q92,150 115,133 Q133,118 155,108" dur="10s" repeatCount="indefinite" keyTimes="0;0.06;0.28;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.8;0.5;0.25;0;0" keyTimes="0;0.06;0.15;0.24;0.28;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- bottom spine 2 (32,178) -> junction (62,158) -> soma edge (155,108) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M32,178 Q48,168 62,158 Q92,150 115,133 Q133,118 155,108" dur="10s" repeatCount="indefinite" keyTimes="0;0.08;0.28;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.8;0.5;0.25;0;0" keyTimes="0;0.08;0.16;0.24;0.28;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- PHASE 2: Soma summation (stroke glow on ellipse above, keyTimes 0.25-0.43) -->
  <!-- PHASE 3: Hillock charges then SHARP threshold flash (3.5-5s) -->
  <circle cx="247" cy="100" r="6" fill="#22d3ee" opacity="0">
    <animate attributeName="opacity" values="0;0;0.1;0.2;0.4;1;0;0" keyTimes="0;0.38;0.40;0.42;0.44;0.47;0.50;1" dur="10s" repeatCount="indefinite"/>
    <animate attributeName="r" values="6;6;7;8;10;14;6;6" keyTimes="0;0.38;0.40;0.42;0.44;0.47;0.50;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- PHASE 4: Saltatory conduction - signal JUMPS node to node (5-7.2s) -->
  <!-- flash at axon start (x=260) -->
  <circle cx="260" cy="100" r="5" fill="#fff" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.3;0;0" keyTimes="0;0.50;0.51;0.53;0.55;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- flash at Node 1 (x=305, gap between myelin 1 and 2) -->
  <circle cx="305" cy="100" r="5" fill="#fff" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.3;0;0" keyTimes="0;0.55;0.56;0.58;0.60;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- flash at Node 2 (x=355, gap between myelin 2 and 3) -->
  <circle cx="355" cy="100" r="5" fill="#fff" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.3;0;0" keyTimes="0;0.60;0.61;0.63;0.65;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- flash at Node 3 (x=405, gap between myelin 3 and 4) -->
  <circle cx="405" cy="100" r="5" fill="#fff" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.3;0;0" keyTimes="0;0.65;0.66;0.68;0.70;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- flash at axon end (x=478) -->
  <circle cx="478" cy="100" r="5" fill="#fff" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.3;0;0" keyTimes="0;0.70;0.71;0.73;0.75;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- PHASE 5: Signal branches to all 6 boutons (7.5-8.8s) -->
  <!-- top bouton 1: (480,100) -> (512,62) -> (540,38) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M480,100 Q498,78 512,62 Q525,48 540,38" dur="10s" repeatCount="indefinite" keyTimes="0;0.76;0.88;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.75;0.76;0.86;0.88;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- top bouton 2: (480,100) -> (512,62) -> (543,50) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M480,100 Q498,78 512,62 Q527,56 543,50" dur="10s" repeatCount="indefinite" keyTimes="0;0.76;0.88;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.75;0.76;0.86;0.88;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- middle bouton 1: (480,100) -> (522,98) -> (553,78) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M480,100 Q505,100 522,98 Q538,88 553,78" dur="10s" repeatCount="indefinite" keyTimes="0;0.76;0.88;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.75;0.76;0.86;0.88;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- middle bouton 2: (480,100) -> (522,98) -> (553,118) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M480,100 Q505,100 522,98 Q538,108 553,118" dur="10s" repeatCount="indefinite" keyTimes="0;0.76;0.88;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.75;0.76;0.86;0.88;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- bottom bouton 1: (480,100) -> (512,138) -> (540,160) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M480,100 Q498,122 512,138 Q525,150 540,160" dur="10s" repeatCount="indefinite" keyTimes="0;0.76;0.88;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.75;0.76;0.86;0.88;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- bottom bouton 2: (480,100) -> (512,138) -> (543,152) -->
  <circle r="3" fill="#fff" opacity="0">
    <animateMotion path="M480,100 Q498,122 512,138 Q527,146 543,152" dur="10s" repeatCount="indefinite" keyTimes="0;0.76;0.88;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0.8;0;0" keyTimes="0;0.75;0.76;0.86;0.88;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- PHASE 6: Bouton flash + vesicle release (8.8-9.5s) -->
  <!-- top bouton 1 flash -->
  <circle cx="540" cy="38" r="4" fill="#fbbf24" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.4;0;0" keyTimes="0;0.87;0.88;0.91;0.94;1" dur="10s" repeatCount="indefinite"/>
    <animate attributeName="r" values="4;4;6;4;4" keyTimes="0;0.87;0.89;0.93;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- middle bouton 1 flash -->
  <circle cx="553" cy="78" r="4" fill="#fbbf24" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.4;0;0" keyTimes="0;0.87;0.88;0.91;0.94;1" dur="10s" repeatCount="indefinite"/>
    <animate attributeName="r" values="4;4;6;4;4" keyTimes="0;0.87;0.89;0.93;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- bottom bouton 1 flash -->
  <circle cx="540" cy="160" r="4" fill="#fbbf24" opacity="0">
    <animate attributeName="opacity" values="0;0;0.9;0.4;0;0" keyTimes="0;0.87;0.88;0.91;0.94;1" dur="10s" repeatCount="indefinite"/>
    <animate attributeName="r" values="4;4;6;4;4" keyTimes="0;0.87;0.89;0.93;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- vesicle dot: top bouton -> across synaptic cleft -->
  <circle r="1.5" fill="#fbbf24" opacity="0">
    <animateMotion path="M540,38 L548,37" dur="10s" repeatCount="indefinite" keyTimes="0;0.90;0.96;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0;0" keyTimes="0;0.89;0.90;0.96;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- vesicle dot: middle bouton -> across synaptic cleft -->
  <circle r="1.5" fill="#fbbf24" opacity="0">
    <animateMotion path="M553,78 L561,77" dur="10s" repeatCount="indefinite" keyTimes="0;0.90;0.96;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0;0" keyTimes="0;0.89;0.90;0.96;1" dur="10s" repeatCount="indefinite"/>
  </circle>
  <!-- vesicle dot: bottom bouton -> across synaptic cleft -->
  <circle r="1.5" fill="#fbbf24" opacity="0">
    <animateMotion path="M540,160 L548,161" dur="10s" repeatCount="indefinite" keyTimes="0;0.90;0.96;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.8;0;0" keyTimes="0;0.89;0.90;0.96;1" dur="10s" repeatCount="indefinite"/>
  </circle>
</svg>

*Figure 1: Complete neuron anatomy and signal mechanics. Graded potentials originate at dendritic spines (green circles) and decay as they passively propagate toward the soma. The soma integrates all inputs (stroke glow). If the summed voltage crosses threshold, the axon hillock (tapered cyan funnel with Na+ channel density marks) fires sharply. The action potential then jumps node-to-node along the myelinated axon via saltatory conduction (each Node of Ranvier flashes at equal intensity). At the synaptic boutons (yellow circles), Ca2+ influx triggers vesicle release across the synaptic cleft (faint green postsynaptic membrane) to the next neuron.*

**Dendrites** are the input structures. They branch outward from the cell body like tree roots, forming a dense receiving network. A single neuron can have thousands of dendritic branches, each receiving signals from different upstream neurons. The point where an upstream neuron's axon terminal meets a dendrite is called a **synapse**, and it is the fundamental unit of neural communication.

**The soma** (cell body) is the integration center. It contains the nucleus and the metabolic machinery that keeps the cell alive, but its computational role is to combine all incoming signals from the dendrites. The soma does not simply add these signals. It integrates them over time and space, with signals arriving at different moments and different locations on the dendritic tree contributing differently to the total.

**The axon hillock** sits at the junction between the soma and the axon. This small region has the highest concentration of voltage-gated sodium channels in the entire neuron, making it the trigger zone. When the integrated signal at the hillock exceeds a threshold (approximately -55mV from a resting potential of -70mV), the neuron fires an action potential.

**The axon** is the output fiber. It carries the action potential away from the soma toward other neurons. Axons can be extremely long, up to a meter in motor neurons running from the spinal cord to the foot. Many axons are wrapped in a fatty insulating layer called **myelin sheath**, with periodic gaps called **Nodes of Ranvier** where the signal is regenerated. This insulation dramatically increases signal speed.

**Axon terminals** (synaptic boutons) are the branching endpoints of the axon. When an action potential arrives, the terminal releases neurotransmitter molecules into the synaptic gap, carrying the signal to the next neuron's dendrites.

## The Action Potential: All-or-Nothing

The action potential is the neuron's signal. Understanding how it works reveals why McCulloch and Pitts chose a binary threshold model.

At rest, the neuron's interior sits at approximately -70 millivolts relative to the outside, maintained by ion pumps that actively transport sodium (Na+) out and potassium (K+) in. This is the **resting potential**.

When excitatory signals from dendrites depolarize the axon hillock past the threshold (around -55mV), voltage-gated sodium channels snap open. Na+ rushes in, driving the voltage sharply positive (to about +40mV). This rapid depolarization is the rising phase.

Within a millisecond, sodium channels inactivate and potassium channels open. K+ flows out, repolarizing the cell back past resting potential to about -80mV (the undershoot or hyperpolarization). The ion pumps then restore the resting state.

<svg viewBox="0 0 420 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Graph of an action potential showing membrane voltage over time: resting at -70mV, rapid depolarization to +40mV peak, repolarization, undershoot to -80mV, then return to resting potential." style="max-width:460px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <title>Action potential voltage trace showing the all-or-nothing spike from resting potential through depolarization, peak, repolarization, and undershoot</title>
  <rect width="420" height="220" rx="8" fill="#181818"/>
  <text x="210" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">action potential</text>
  <!-- axes -->
  <line x1="60" y1="30" x2="60" y2="190" stroke="#555" stroke-width="1"/>
  <line x1="60" y1="190" x2="400" y2="190" stroke="#555" stroke-width="1"/>
  <!-- Y axis labels -->
  <text x="55" y="48" text-anchor="end" fill="#999" font-family="monospace" font-size="8">+40mV</text>
  <text x="55" y="108" text-anchor="end" fill="#999" font-family="monospace" font-size="8">-55mV</text>
  <text x="55" y="140" text-anchor="end" fill="#999" font-family="monospace" font-size="8">-70mV</text>
  <text x="55" y="168" text-anchor="end" fill="#999" font-family="monospace" font-size="8">-80mV</text>
  <!-- X axis label -->
  <text x="230" y="210" text-anchor="middle" fill="#999" font-family="monospace" font-size="9">time (ms)</text>
  <!-- threshold line -->
  <line x1="60" y1="105" x2="400" y2="105" stroke="#fbbf24" stroke-width="0.5" stroke-dasharray="4"/>
  <text x="402" y="108" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="start">threshold</text>
  <!-- resting potential line -->
  <line x1="60" y1="137" x2="400" y2="137" stroke="#4ade80" stroke-width="0.5" stroke-dasharray="4"/>
  <!-- action potential trace -->
  <path d="M60,137 L120,137 L140,137 Q155,135 165,105 Q175,60 180,45 Q185,55 190,85 Q200,145 205,165 Q210,170 220,145 L240,137 L400,137" fill="none" stroke="#22d3ee" stroke-width="2"/>
  <!-- phase labels -->
  <text x="100" y="130" fill="#4ade80" font-family="monospace" font-size="7" text-anchor="middle">resting</text>
  <text x="150" y="95" fill="#999" font-family="monospace" font-size="7" text-anchor="middle" transform="rotate(-70,150,95)">depolarize</text>
  <text x="183" y="36" fill="#22d3ee" font-family="monospace" font-size="7" text-anchor="middle">peak</text>
  <text x="200" y="130" fill="#999" font-family="monospace" font-size="7" text-anchor="middle" transform="rotate(70,200,130)">repolarize</text>
  <text x="215" y="178" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">undershoot</text>
  <!-- animated dot tracing the action potential -->
  <circle r="4" fill="#fff" opacity="0">
    <animateMotion path="M60,137 L120,137 L140,137 Q155,135 165,105 Q175,60 180,45 Q185,55 190,85 Q200,145 205,165 Q210,170 220,145 L240,137 L400,137" dur="4s" repeatCount="indefinite" keyTimes="0;0.1;0.2;1" keyPoints="0;0;0.2;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.09;0.1;0.9;0.95;1" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>

*Figure 2: The action potential. Membrane voltage sits at the resting potential (-70mV) until depolarization crosses the threshold (-55mV), triggering a rapid spike to +40mV. The signal is all-or-nothing: same amplitude every time, regardless of stimulus strength.*

The critical insight: this is a **binary event**. The neuron either fires a full action potential or it does not fire at all. There is no partial spike, no half-signal. The amplitude of every action potential is the same regardless of stimulus strength. Stimulus intensity is encoded in **firing rate** (how many spikes per second), not spike amplitude.

This all-or-nothing property is what McCulloch and Pitts latched onto. It maps directly to a binary output: 1 or 0, fire or do not fire.

## Synaptic Transmission: Weights Before There Were Weights

Neurons communicate across synapses. When an action potential reaches an axon terminal, it triggers the release of neurotransmitter molecules into the synaptic cleft (the gap between neurons, roughly 20 nanometers wide). These molecules bind to receptors on the receiving neuron's dendrite, opening ion channels that either depolarize or hyperpolarize the receiving cell.

This is where the biology maps most directly to the mathematics of neural networks.

**Excitatory synapses** (using neurotransmitters like glutamate) push the receiving neuron toward firing. They depolarize the membrane, bringing it closer to threshold. In the mathematical model, these correspond to **positive weights**.

**Inhibitory synapses** (using neurotransmitters like GABA) push the receiving neuron away from firing. They hyperpolarize the membrane, making it harder to reach threshold. These correspond to **negative weights**.

Not all synapses are equal. Some connections are strong (more neurotransmitter released, more receptors available, larger effect on membrane potential) and some are weak. The strength of a synapse determines how much influence it has on the receiving neuron's decision to fire. In mathematical terms, this is exactly what a weight does: it scales the input.

Furthermore, synaptic strength is not fixed. **Synaptic plasticity**, the ability of synapses to strengthen or weaken over time, is the biological basis of learning. Donald Hebb captured this principle in 1949: "neurons that fire together wire together." When a presynaptic neuron consistently contributes to causing the postsynaptic neuron to fire, the synapse between them strengthens. This anticipates the concept of adjustable weights in artificial neural networks by over a decade.

## Spatial and Temporal Integration

The soma does not evaluate each incoming signal independently. It performs two types of integration.

**Spatial summation** occurs when signals arrive from multiple synapses at the same time. Each synapse produces a small voltage change (an excitatory or inhibitory postsynaptic potential). These changes propagate through the dendrites to the soma, where they combine. If enough excitatory inputs arrive simultaneously to push the hillock past threshold, the neuron fires. This is the biological analog of the weighted sum: multiply each input by its synaptic strength and add them up.

**Temporal summation** occurs when signals arrive from the same synapse in rapid succession. Each individual signal might be too weak to trigger firing, but if they arrive fast enough, the voltage changes accumulate before the previous one decays. This integrates information over time, a property that McCulloch and Pitts deliberately excluded from their model.

<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg" style="max-width:540px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="500" height="180" rx="8" fill="#181818"/>
  <text x="250" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">signal integration at the soma</text>
  <!-- three input signals arriving -->
  <!-- signal 1: excitatory (strong) -->
  <rect x="20" y="35" width="80" height="25" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="60" y="51" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">excitatory +3</text>
  <!-- signal 2: excitatory (weak) -->
  <rect x="20" y="70" width="80" height="25" rx="4" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="60" y="86" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">excitatory +1</text>
  <!-- signal 3: inhibitory -->
  <rect x="20" y="105" width="80" height="25" rx="4" fill="#333" stroke="#ef4444" stroke-width="1"/>
  <text x="60" y="121" fill="#ef4444" font-family="monospace" font-size="8" text-anchor="middle">inhibitory -2</text>
  <!-- arrows to soma -->
  <line x1="100" y1="47" x2="155" y2="78" stroke="#4ade80" stroke-width="1"/>
  <line x1="100" y1="82" x2="155" y2="82" stroke="#4ade80" stroke-width="1"/>
  <line x1="100" y1="117" x2="155" y2="88" stroke="#ef4444" stroke-width="1"/>
  <!-- soma integration -->
  <ellipse cx="190" cy="82" rx="35" ry="25" fill="#333" stroke="#888" stroke-width="1.5"/>
  <text x="190" y="79" fill="#ccc" font-family="monospace" font-size="9" text-anchor="middle">sum</text>
  <text x="190" y="92" fill="#22d3ee" font-family="monospace" font-size="10" text-anchor="middle">= +2</text>
  <!-- threshold comparison -->
  <line x1="225" y1="82" x2="280" y2="82" stroke="#888" stroke-width="1.5"/>
  <rect x="280" y="62" width="70" height="40" rx="4" fill="#333" stroke="#fbbf24" stroke-width="1"/>
  <text x="315" y="79" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">threshold</text>
  <text x="315" y="94" fill="#fbbf24" font-family="monospace" font-size="10" text-anchor="middle">= +2</text>
  <!-- output -->
  <line x1="350" y1="82" x2="400" y2="82" stroke="#888" stroke-width="1.5"/>
  <rect x="400" y="62" width="70" height="40" rx="4" fill="#333" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="435" y="79" fill="#22d3ee" font-family="monospace" font-size="9" text-anchor="middle">FIRE</text>
  <text x="435" y="94" fill="#22d3ee" font-family="monospace" font-size="10" text-anchor="middle">y = 1</text>
  <!-- animated pulse showing summation -->
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M100,47 L155,78 L190,82" dur="3s" repeatCount="indefinite" keyTimes="0;0.1;0.3;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.1;0.28;0.3;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M100,82 L155,82 L190,82" dur="3s" repeatCount="indefinite" keyTimes="0;0.15;0.35;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.14;0.15;0.33;0.35;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#ef4444" opacity="0">
    <animateMotion path="M100,117 L155,88 L190,82" dur="3s" repeatCount="indefinite" keyTimes="0;0.2;0.4;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.19;0.2;0.38;0.4;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- output pulse -->
  <circle r="4" fill="#fff" opacity="0">
    <animateMotion path="M225,82 L315,82 L435,82" dur="3s" repeatCount="indefinite" keyTimes="0;0.5;0.8;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.49;0.5;0.78;0.8;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- explanation -->
  <text x="250" y="170" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">+3 + 1 + (-2) = +2 >= threshold -> neuron fires</text>
</svg>

## The McCulloch-Pitts Neuron (1943)

Warren McCulloch was a neurophysiologist. Walter Pitts was a mathematical logician. In 1943, they published "A Logical Calculus of the Ideas Immanent in Nervous Activity," proposing the first mathematical model of a neuron.

Their approach was deliberately reductive. They started with the biological neuron and asked: what is the minimum abstraction that preserves the computational behavior?

### What They Kept

1. **Multiple inputs**: A neuron receives signals from many sources (dendrites from many upstream neurons).
2. **Weighted combination**: Each input has a different influence (synaptic strength varies).
3. **Threshold firing**: If the combined signal exceeds a threshold, the neuron fires (all-or-nothing).
4. **Binary output**: The neuron either fires (1) or does not fire (0).

### What They Threw Away

1. **Temporal dynamics**: Real neurons integrate signals over time. The McCulloch-Pitts model computes instantaneously.
2. **Analog voltage**: Real membrane potentials are continuous values. The model uses binary.
3. **Spatial structure**: Dendritic geometry matters in real neurons. The model treats all inputs as arriving at a single point.
4. **Refractory period**: Real neurons cannot fire again immediately after an action potential. The model has no memory of previous states.
5. **Neurotransmitter chemistry**: The model reduces the complex molecular machinery of synaptic transmission to a simple multiplication.
6. **Firing rate**: Real neurons encode information in spike frequency. The model produces a single binary value.

### The Mathematical Formulation

The McCulloch-Pitts neuron computes:

```
y = f(w1*x1 + w2*x2 + ... + wn*xn)
```

Where:
- `xi` are the binary inputs (0 or 1)
- `wi` are the weights (positive for excitatory, negative for inhibitory)
- `f` is the threshold function: output 1 if the sum exceeds the threshold, 0 otherwise

More precisely, with bias term `b` replacing the threshold:

```
y = 1  if  w1*x1 + w2*x2 + ... + wn*xn + b >= 0
y = 0  otherwise
```

The bias `b` is the negative of the threshold. Moving it to the left side of the inequality recovers the original formulation: fire if the weighted sum exceeds the threshold.

<svg viewBox="0 0 440 150" xmlns="http://www.w3.org/2000/svg" style="max-width:520px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="440" height="150" rx="8" fill="#181818"/>
  <text x="220" y="18" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">McCulloch-Pitts neuron</text>
  <!-- input nodes -->
  <circle cx="40" cy="35" r="14" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="39" fill="#ccc" font-family="monospace" font-size="11" text-anchor="middle">x1</text>
  <circle cx="40" cy="75" r="14" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="79" fill="#ccc" font-family="monospace" font-size="11" text-anchor="middle">x2</text>
  <circle cx="40" cy="115" r="14" fill="#333" stroke="#4ade80" stroke-width="1.5"/>
  <text x="40" y="119" fill="#ccc" font-family="monospace" font-size="11" text-anchor="middle">xn</text>
  <!-- weight labels -->
  <text x="105" y="30" fill="#999" font-family="monospace" font-size="9" text-anchor="middle">w1</text>
  <text x="105" y="72" fill="#999" font-family="monospace" font-size="9" text-anchor="middle">w2</text>
  <text x="105" y="112" fill="#999" font-family="monospace" font-size="9" text-anchor="middle">wn</text>
  <!-- edges -->
  <line x1="54" y1="35" x2="156" y2="70" stroke="#555" stroke-width="1.5"/>
  <line x1="54" y1="75" x2="156" y2="75" stroke="#555" stroke-width="1.5"/>
  <line x1="54" y1="115" x2="156" y2="80" stroke="#555" stroke-width="1.5"/>
  <!-- signal pulses -->
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M54,35 L156,70" dur="3s" repeatCount="indefinite" keyTimes="0;0.3;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.28;0.3;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M54,75 L156,75" dur="3s" repeatCount="indefinite" keyTimes="0;0.3;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.28;0.3;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#4ade80" opacity="0">
    <animateMotion path="M54,115 L156,80" dur="3s" repeatCount="indefinite" keyTimes="0;0.3;1" keyPoints="0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0.9;0.9;0;0" keyTimes="0;0.01;0.28;0.3;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- sum node -->
  <circle cx="170" cy="75" r="18" fill="#333" stroke="#888" stroke-width="1.5">
    <animate attributeName="stroke" values="#888;#888;#22d3ee;#22d3ee;#888;#888" keyTimes="0;0.29;0.31;0.5;0.52;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="170" y="79" fill="#ccc" font-family="monospace" font-size="10" text-anchor="middle">sum</text>
  <!-- edge to threshold -->
  <line x1="188" y1="75" x2="250" y2="75" stroke="#555" stroke-width="1.5"/>
  <!-- threshold pulse -->
  <circle r="4" fill="#fff" opacity="0">
    <animateMotion path="M188,75 L250,75" dur="3s" repeatCount="indefinite" keyTimes="0;0.5;0.7;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0;0" keyTimes="0;0.49;0.5;0.68;0.7;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- threshold node -->
  <circle cx="265" cy="75" r="18" fill="#333" stroke="#fbbf24" stroke-width="1.5">
    <animate attributeName="stroke" values="#888;#888;#fbbf24;#fbbf24;#888;#888" keyTimes="0;0.69;0.71;0.82;0.84;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="265" y="72" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">>=</text>
  <text x="265" y="83" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">thr</text>
  <!-- edge to output -->
  <line x1="283" y1="75" x2="365" y2="75" stroke="#555" stroke-width="1.5"/>
  <!-- output pulse -->
  <circle r="4" fill="#fbbf24" opacity="0">
    <animateMotion path="M283,75 L365,75" dur="3s" repeatCount="indefinite" keyTimes="0;0.82;0.97;1" keyPoints="0;0;1;1" calcMode="linear"/>
    <animate attributeName="opacity" values="0;0;0.9;0.9;0" keyTimes="0;0.81;0.82;0.96;0.97" dur="3s" repeatCount="indefinite"/>
  </circle>
  <!-- output node -->
  <circle cx="380" cy="75" r="14" fill="#333" stroke="#888" stroke-width="1.5">
    <animate attributeName="stroke" values="#888;#888;#fbbf24;#fbbf24;#888" keyTimes="0;0.96;0.97;0.99;1" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="380" y="79" fill="#ccc" font-family="monospace" font-size="11" text-anchor="middle">y</text>
  <!-- labels -->
  <text x="40" y="143" fill="#4ade80" font-family="monospace" font-size="8" text-anchor="middle">inputs</text>
  <text x="170" y="143" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">weighted sum</text>
  <text x="265" y="143" fill="#fbbf24" font-family="monospace" font-size="8" text-anchor="middle">threshold</text>
  <text x="380" y="143" fill="#999" font-family="monospace" font-size="8" text-anchor="middle">output</text>
</svg>

## Computing with Threshold Logic

McCulloch and Pitts did not just propose a model. They proved something profound: networks of their idealized neurons can compute any logical function. A single McCulloch-Pitts neuron can implement the basic Boolean logic gates.

### AND Gate

An AND gate outputs 1 only when both inputs are 1. Set both weights to 1 and the threshold to 2:

```
weights: w1 = 1, w2 = 1
threshold: 2

x1=0, x2=0 -> 0+0 = 0 < 2 -> output 0
x1=1, x2=0 -> 1+0 = 1 < 2 -> output 0
x1=0, x2=1 -> 0+1 = 1 < 2 -> output 0
x1=1, x2=1 -> 1+1 = 2 >= 2 -> output 1
```

### OR Gate

An OR gate outputs 1 when at least one input is 1. Set both weights to 1 and the threshold to 1:

```
weights: w1 = 1, w2 = 1
threshold: 1

x1=0, x2=0 -> 0+0 = 0 < 1 -> output 0
x1=1, x2=0 -> 1+0 = 1 >= 1 -> output 1
x1=0, x2=1 -> 0+1 = 1 >= 1 -> output 1
x1=1, x2=1 -> 1+1 = 2 >= 1 -> output 1
```

### NOT Gate

A NOT gate inverts the input. Set the weight to -1 and the threshold to 0:

```
weight: w1 = -1
threshold: 0 (bias = 0)

x1=0 -> -1*0 = 0 >= 0 -> output 1
x1=1 -> -1*1 = -1 < 0 -> output 0
```

<svg viewBox="0 0 500 160" xmlns="http://www.w3.org/2000/svg" style="max-width:540px;width:100%;height:auto;display:block;margin:1.5em auto;">
  <rect width="500" height="160" rx="8" fill="#181818"/>
  <text x="250" y="16" text-anchor="middle" fill="#999" font-family="monospace" font-size="10">logic gates as McCulloch-Pitts neurons</text>
  <!-- AND gate -->
  <text x="83" y="35" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9" font-weight="bold">AND</text>
  <circle cx="30" cy="60" r="10" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="30" y="64" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">x1</text>
  <circle cx="30" cy="100" r="10" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="30" y="104" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">x2</text>
  <line x1="40" y1="60" x2="70" y2="75" stroke="#555" stroke-width="1"/>
  <line x1="40" y1="100" x2="70" y2="85" stroke="#555" stroke-width="1"/>
  <text x="55" y="56" fill="#999" font-family="monospace" font-size="7">1</text>
  <text x="55" y="104" fill="#999" font-family="monospace" font-size="7">1</text>
  <circle cx="83" cy="80" r="14" fill="#333" stroke="#888" stroke-width="1.5"/>
  <text x="83" y="77" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">t=2</text>
  <text x="83" y="87" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">f</text>
  <line x1="97" y1="80" x2="130" y2="80" stroke="#555" stroke-width="1"/>
  <circle cx="140" cy="80" r="10" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="140" y="84" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">y</text>
  <text x="83" y="120" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">1 AND 1 = 1</text>
  <text x="83" y="132" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">1 AND 0 = 0</text>
  <text x="83" y="144" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">0 AND 0 = 0</text>
  <!-- OR gate -->
  <text x="250" y="35" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9" font-weight="bold">OR</text>
  <circle cx="197" cy="60" r="10" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="197" y="64" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">x1</text>
  <circle cx="197" cy="100" r="10" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="197" y="104" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">x2</text>
  <line x1="207" y1="60" x2="237" y2="75" stroke="#555" stroke-width="1"/>
  <line x1="207" y1="100" x2="237" y2="85" stroke="#555" stroke-width="1"/>
  <text x="222" y="56" fill="#999" font-family="monospace" font-size="7">1</text>
  <text x="222" y="104" fill="#999" font-family="monospace" font-size="7">1</text>
  <circle cx="250" cy="80" r="14" fill="#333" stroke="#888" stroke-width="1.5"/>
  <text x="250" y="77" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">t=1</text>
  <text x="250" y="87" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">f</text>
  <line x1="264" y1="80" x2="297" y2="80" stroke="#555" stroke-width="1"/>
  <circle cx="307" cy="80" r="10" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="307" y="84" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">y</text>
  <text x="250" y="120" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">1 OR 0 = 1</text>
  <text x="250" y="132" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">0 OR 1 = 1</text>
  <text x="250" y="144" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">0 OR 0 = 0</text>
  <!-- NOT gate -->
  <text x="417" y="35" text-anchor="middle" fill="#22d3ee" font-family="monospace" font-size="9" font-weight="bold">NOT</text>
  <circle cx="370" cy="80" r="10" fill="#333" stroke="#4ade80" stroke-width="1"/>
  <text x="370" y="84" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">x1</text>
  <line x1="380" y1="80" x2="403" y2="80" stroke="#555" stroke-width="1"/>
  <text x="392" y="74" fill="#ef4444" font-family="monospace" font-size="7">-1</text>
  <circle cx="417" cy="80" r="14" fill="#333" stroke="#888" stroke-width="1.5"/>
  <text x="417" y="77" fill="#fbbf24" font-family="monospace" font-size="7" text-anchor="middle">t=0</text>
  <text x="417" y="87" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">f</text>
  <line x1="431" y1="80" x2="457" y2="80" stroke="#555" stroke-width="1"/>
  <circle cx="467" cy="80" r="10" fill="#333" stroke="#888" stroke-width="1"/>
  <text x="467" y="84" fill="#ccc" font-family="monospace" font-size="8" text-anchor="middle">y</text>
  <text x="417" y="120" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">NOT 1 = 0</text>
  <text x="417" y="132" fill="#999" font-family="monospace" font-size="7" text-anchor="middle">NOT 0 = 1</text>
</svg>

### Toward Turing-Completeness

Since AND, OR, and NOT gates are sufficient to compute any Boolean function (they form a functionally complete set), McCulloch and Pitts showed that networks of their idealized neurons can, in principle, compute anything that Boolean circuits can compute.

They went further. By adding feedback loops (outputs feeding back as inputs, introducing a notion of time steps), they argued that networks of binary threshold units can simulate any finite automaton. This was a remarkable result: it linked neural computation to the formal theory of computation that Turing had developed just seven years earlier.

The practical limitation was glaring: the weights had to be **set by hand**. McCulloch and Pitts provided no mechanism for a network to learn the right weights from data. Their model was a proof of computational universality, not a learning algorithm. The question of how to find the right weights would take another 15 years to answer.

## Biology vs. Model: A Side-by-Side View

Understanding what the McCulloch-Pitts model captures and what it misses is essential for understanding every subsequent development in neural networks. Each generation of models recovered some piece of biology that McCulloch and Pitts discarded.

| Biological Property | McCulloch-Pitts | Later Models |
|---|---|---|
| Multiple inputs | Yes | Yes |
| Variable connection strength | Yes (weights) | Yes (learnable weights) |
| Threshold firing | Yes (step function) | Yes (activation functions) |
| Binary output | Yes | No (continuous activations) |
| Temporal dynamics | No | RNNs, LSTMs |
| Analog voltage | No | Continuous-valued neurons |
| Spatial dendritic structure | No | Still mostly ignored |
| Synaptic plasticity | No | Backpropagation |
| Refractory period | No | Spiking neural networks |
| Firing rate coding | No | Rate-coded networks |
| Neurotransmitter diversity | No | Still mostly ignored |

The original model was deliberately minimal. It captured the essence of neural computation, enough to prove universality, but not enough to learn. The next step in the lineage, the perceptron, would add exactly what was missing: a rule for adjusting the weights automatically.

## Key Takeaways

The biological neuron is a sophisticated electrochemical computer. McCulloch and Pitts stripped it down to its computational essence: weighted inputs, summation, threshold, binary output. That abstraction was powerful enough to prove that networks of simple threshold units can compute any Boolean function. But the weights had to be hand-designed. The path from here to modern deep learning is the story of recovering, piece by piece, the capabilities that this first model threw away.
