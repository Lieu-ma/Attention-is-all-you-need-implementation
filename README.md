# Attention-is-all-you-need-implementation
## PyTorch Transformer Implementation

A from-scratch implementation of the Transformer architecture ("Attention Is All You Need") using PyTorch, featuring multi-head attention, positional encoding, and full encoder-decoder structure.

## Features

- Pure PyTorch implementation (no external dependencies beyond PyTorch)
- Multi-head scaled dot-product attention
- Positional embeddings for sequence order
- Residual connections and layer normalization
- Padding and look-ahead masking
- Modular design for easy customization
- GPU support via CUDA

## Architecture Components

### 1. SelfAttention
- Implements multi-head attention mechanism
- Handles queries, keys, and values projections
- Includes attention masking and scaling

### 2. TransformerBlock
- Self-attention + feed-forward network
- Residual connections and layer norm
- Dropout for regularization

### 3. Encoder
- Stack of Transformer blocks
- Token and positional embeddings
- Source padding mask

### 4. Decoder
- Masked self-attention
- Encoder-decoder attention
- Output probability prediction

## Installation

```bash
pip install torch
