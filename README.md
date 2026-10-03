# BEM–Graph Neural Network Models for Boundary Reconstruction

This repository contains two related graph neural network implementations for boundary reconstruction in two-dimensional Boundary Element Method (BEM) problems. Both models use full-graph attention to capture nonlocal interactions among boundary elements and combine physics-based BEM computation with data-driven graph learning.

## Repository Overview

```text
.
├── Single_Output_Attention_Model/
│   └── Full-graph attention VAE for boundary flux prediction
│
└── Dual_Output_Attention_Model/
    └── Dual-decoder full-graph attention VAE for simultaneous boundary-field reconstruction
```

### 1. Single-Output Attention Model

The first implementation uses a shared full-graph attention encoder and a single decoder to predict the boundary flux \(q\).

The main architecture is:

```text
Boundary Features
      ↓
Full-Graph Attention Encoder
      ↓
Variational Latent Representation
      ↓
Full-Graph Attention Decoder
      ↓
Boundary Flux q
```

The predicted boundary flux can subsequently be used by the BEM solver to reconstruct the interior field.

### 2. Dual-Output Attention Model

The second implementation extends the attention-based framework to mixed boundary conditions. A shared encoder extracts the boundary representation, followed by two independent decoders for reconstructing the two boundary quantities.

```text
Boundary Features
      ↓
Full-Graph Attention Encoder
      ↓
Variational Latent Representation
      ↓
   ┌──┴──┐
   ↓     ↓
T Decoder   q Decoder
   ↓         ↓
   T         q
```

This design allows Dirichlet and Neumann boundary quantities to be reconstructed within the same graph-learning framework while retaining separate decoder parameters for the two physical fields.

## Common Framework

Both implementations share the same general methodology:

* Boundary elements are represented as graph nodes.
* Fully connected graphs model nonlocal boundary interactions.
* Multi-head graph attention learns interactions between boundary elements.
* Variational latent representations are used in the encoder-decoder architecture.
* BEM provides the physics-based boundary and interior-field computations.
* Neural predictions are evaluated against reference BEM or analytical solutions.

The two implementations mainly differ in their prediction objectives: the first model focuses on a **single boundary-flux output**, whereas the second model introduces **dual decoder branches for two boundary quantities**.

Each subdirectory contains its own source code, training and evaluation scripts, dependency information, and detailed documentation.
