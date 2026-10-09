## **Chapter 02: LLM Architectures**, covering Transformers, Scaled Dot-Product Attention, Quantization (GGUF, AWQ), and Context Windows.

---

## 1. Transformer Architecture & Attention Mechanisms

Modern Large Language Models (LLMs) are built on decoder-only Transformer architectures (e.g., Llama, Mistral, Qwen) optimized for causal auto-regressive next-token prediction.

```
Input Tokens ──► Embedding + Positional Encoding ──► [ Multi-Head Attention / RoPE ──► Feed Forward (SwiGLU) ] x N ──► Softmax ──► Next Token

```

### Scaled Dot-Product Attention

Attention transforms input sequence embeddings into contextual representations by calculating dynamic affinity scores between query ($Q$), key ($K$), and value ($V$) projections:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

* **$Q, K, V$ Projections:** $Q = X W_Q$, $K = X W_K$, $V = X W_V$ where $X \in \mathbb{R}^{T \times d_{\text{model}}}$.
* **Scaling Factor $\frac{1}{\sqrt{d_k}}$:** Prevents dot-product values from growing extremely large in higher dimensions ($d_k$), which would force the `softmax` function into regions with near-zero gradients.
* **Causal Mask ($M$):** An upper-triangular matrix filled with $-\infty$ above the diagonal to ensure tokens can only attend to previous tokens ($t' \le t$).

### Multi-Query (MQA) & Grouped-Query Attention (GQA)

Standard Multi-Head Attention (MHA) maintains separate Key-Value ($KV$) heads for every Query ($Q$) head, creating massive memory bandwidth bottlenecks during inference.

```
MHA (8 Q, 8 KV)          GQA (8 Q, 2 KV Groups)       MQA (8 Q, 1 KV)
 Q Q Q Q Q Q Q Q          Q Q Q Q  Q Q Q Q             Q Q Q Q Q Q Q Q
 | | | | | | | |           \  /     \  /                \ \ \ | / / /
 K K K K K K K K            K        K                        K
 V V V V V V V V            V        V                        V

```

* **Multi-Head Attention (MHA):** $N$ Query heads, $N$ $KV$ heads.
* **Grouped-Query Attention (GQA):** $N$ Query heads grouped across $G$ shared $KV$ heads (e.g., 8 Queries per $1$ $KV$ head).
* **Multi-Query Attention (MQA):** $N$ Query heads share a single $KV$ head.

---

## 2. Context Windows & Position Encodings

Because standard Transformers lose sequence order information, positional encodings are added to token embeddings.

### Rotary Position Embedding (RoPE)

RoPE applies a complex rotation matrix to Query and Key vectors based on their token index $m$, allowing models to naturally compute relative positional distances:

$$R_{\Theta, m}^d = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 \end{pmatrix}$$

### Context Window Scaling Techniques

* **Position Interpolation (PI):** Linear downscaling of position indices ($m' = m / S$) to fit longer sequences into a pre-trained context window.
* **YaRN (Yet another RoPE extensioN):** Applies non-uniform scaling across different frequency dimensions, preserving high-frequency details for short distances while expanding low frequencies for long-context stability.

---

## 3. LLM Quantization Frameworks

Quantization compresses model weights from 16-bit floating point (`FP16`/`BF16`) down to 8-bit, 4-bit, or lower integers.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LLM Quantization Formats                        │
├───────────────────────────────┬────────────────────────────────────────┤
│ GGUF                          │ AWQ / GPTQ                             │
│ • File Container (.gguf)      │ • Safetensors + PTQ Algorithm          │
│ • Runs on CPU, GPU, Apple     │ • Optimized for vLLM & TensorRT        │
│ • Ideal for local & edge      │ • Production GPU Server Deployment     │
└───────────────────────────────┴────────────────────────────────────────┘

```

### AWQ (Activation-Aware Weight Quantization)

AWQ protects the top ~1% of weights corresponding to high-magnitude activation features before applying uniform integer quantization:

1. Observes activation distributions on calibration datasets.
2. Identifies critical weights ("salient weights") that directly impact model output.
3. Scales these channels up prior to 4-bit integer conversion, minimizing quantization error without per-weight optimization loops.

### GGUF (GGML Unified Format)

GGUF is a single-file binary container format designed for `llama.cpp` and local runtimes (e.g., Ollama, LM Studio):

* Stores weights, tokenizer metadata, and architecture hyper-parameters in one file.
* Utilizes **K-Quants** (e.g., `Q4_K_M`, `Q5_K_M`) which dynamically assign varying bit-widths across layers based on sensitivity.
* Supports hybrid layer offloading between CPU RAM and GPU VRAM.

---

## 4. Python Implementation: Scaled Dot-Product & Quantization
Below is a complete PyTorch implementation demonstrating causal scaled dot-product attention and a basic uniform 4-bit weight quantizer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Scaled Dot-Product Attention with Causal Masking
def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Args:
        query: Shape (Batch, Heads, Seq_Len, d_k)
        key:   Shape (Batch, Heads, Seq_Len, d_k)
        value: Shape (Batch, Heads, Seq_Len, d_v)
        mask:  Optional Causal Mask
    """
    d_k = query.size(-1)
    
    # Compute dot product affinity matrix: (B, H, T, d_k) x (B, H, d_k, T) -> (B, H, T, T)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output

# 2. Simple Uniform 4-Bit Symmetric Quantizer
def quantize_weights_4bit(weights: torch.Tensor):
    """
    Simulates INT4 symmetric weight quantization.
    """
    max_val = torch.max(torch.abs(weights))
    scale = max_val / 7.0  # INT4 symmetric signed range: [-7, 7]
    
    # Quantize to integer steps
    quantized_int = torch.round(weights / scale).clamp(-7, 7)
    
    # Dequantize back for FP16 execution
    dequantized_weights = quantized_int * scale
    return quantized_int.to(torch.int8), scale, dequantized_weights


# Verification
batch_size, num_heads, seq_len, d_k = 1, 4, 8, 16
q = torch.randn(batch_size, num_heads, seq_len, d_k)
k = torch.randn(batch_size, num_heads, seq_len, d_k)
v = torch.randn(batch_size, num_heads, seq_len, d_k)

# Create causal mask (lower triangular)
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

# Run Attention
attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
print("Attention Output Shape:", attn_output.shape)

# Run Quantization Simulation
sample_weights = torch.randn(128, 128)
q_int8, scale, dequant_w = quantize_weights_4bit(sample_weights)
print("Quantization MSE Error:", F.mse_loss(sample_weights, dequant_w).item())
```
---

Here is a detailed breakdown of how each component of the code snippet works, covering both the **Causal Scaled Dot-Product Attention** mechanism and the **4-Bit Weight Quantization** simulator.

---

### Part 1: Scaled Dot-Product Attention (`scaled_dot_product_attention`)

```python
def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:

```

#### Step-by-Step Breakdown:

1. **Extract Projection Dimension (`d_k`):**
```python
d_k = query.size(-1)

```


* Retrieves $d_k$ (the size of each attention head). This dimension is used to compute the scaling factor $\sqrt{d_k}$.


2. **Compute Unscaled Attention Scores (Matrix Multiplication):**
```python
scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

```


* **`key.transpose(-2, -1)`:** Transposes the last two dimensions of Key from `(Batch, Heads, Seq_Len, d_k)` to `(Batch, Heads, d_k, Seq_Len)`.
* **`torch.matmul(...)`:** Multiplies $Q$ and $K^T$, outputting a matrix of shape `(Batch, Heads, Seq_Len, Seq_Len)`. Each cell $(i, j)$ represents the raw relevance (dot-product similarity) between token $i$ and token $j$.
* **` / (d_k ** 0.5)`:** Scales down the scores by $\sqrt{d_k}$. Without scaling, dot products grow large in higher dimensions, pushing Softmax into saturated regions with tiny gradients.


3. **Apply Causal Masking:**
```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))

```


* Replaces positions where `mask == 0` (future tokens) with $-\infty$.
* When Softmax is applied, $e^{-\infty} = 0$, guaranteeing that tokens **cannot attend to future tokens** (preserving autoregressive causal integrity).


4. **Convert Scores to Probabilities & Weighted Sum:**
```python
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, value)

```


* **`F.softmax(..., dim=-1)`:** Converts raw logit scores into valid probabilities across rows so that each row sums to $1.0$.
* **`torch.matmul(attn_weights, value)`:** Multiplies the probability weights by the Value tensor $V$. The resulting tensor of shape `(Batch, Heads, Seq_Len, d_k)` represents the contextualized token representations.



---

### Part 2: 4-Bit Symmetric Weight Quantization (`quantize_weights_4bit`)

```python
def quantize_weights_4bit(weights: torch.Tensor):

```

#### Step-by-Step Breakdown:

1. **Find Maximum Absolute Value & Scale Factor:**
```python
max_val = torch.max(torch.abs(weights))
scale = max_val / 7.0  # INT4 symmetric signed range: [-7, 7]

```


* Determines the dynamic range of the layer's weights.
* Signed 4-bit integers (`INT4`) represent values from $-7$ to $+7$ ($2^3 - 1 = 7$). Dividing `max_val` by $7.0$ calculates the **scale step** representing 1 discrete integer level.


2. **Quantize Floating-Point Weights to Integers:**
```python
quantized_int = torch.round(weights / scale).clamp(-7, 7)

```


* **`weights / scale`:** Normalizes raw FP32/FP16 weights into the step range $[-7, 7]$.
* **`torch.round(...)`:** Discretizes values to the nearest whole integer.
* **`clamp(-7, 7)`:** Enforces hard boundary constraints to prevent overflow.


3. **Dequantize Back to Floating-Point:**
```python
dequantized_weights = quantized_int * scale

```


* Reconstructs approximate original weights during matrix multiplication by multiplying integer levels by the original `scale` factor.
* Measuring Mean Squared Error (`MSE`) between `sample_weights` and `dequant_w` quantifies information loss caused by precision reduction.



---

### Code Execution Summary

* **Shape Verification:** Verifies that passing tensors through $Q, K, V$ dimensions returns an output shape identical to input $Q$: `(1, 4, 8, 16)`.
* **Causal Mask Tensor:** `torch.tril(torch.ones(8, 8))` creates an $8 \times 8$ lower-triangular matrix of $1$s and $0$s, ensuring token $i$ only sees tokens $\le i$.
* **MSE Error Logging:** Demonstrates how much weight precision is preserved under 4-bit quantization (typically yielding low MSE for normally distributed linear layer weights).


---

## What is KV-Caching?

During autoregressive generation, generating token $N$ requires computing attention over all preceding tokens ($1$ through $N-1$). Without caching, the model re-computes $K$ and $V$ projections for every previous token at every single step, resulting in $O(N^2)$ redundant matrix operations.

```
Without Cache (Step N):  Compute K & V for Tokens [1, 2, ..., N-1, N]  --> O(N^2)
With Cache (Step N):     Compute K & V ONLY for Token N
                        Append to Cache: [Cache, New_K], [Cache, New_V]  --> O(N)

```

With a KV-Cache:

1. **Prefill Phase (Step 1):** Processes the prompt sequence of length $L$. Computes and stores $K_{1..L}$ and $V_{1..L}$ in memory.
2. **Decode Phase (Steps $2 \dots N$):** Processes only **one new token at a time** (sequence length $1$). Computes its new key ($K_{\text{new}}$) and value ($V_{\text{new}}$), concatenates them onto the existing cache along the sequence dimension, and performs attention over the entire cached sequence length.

---

## Python & PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class KVCache:
    """
    Stores and manages cached Key and Value tensors across generation steps.
    """
    def __init__(self):
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new key/value states to the cache along the sequence dimension (dim=-2).
        
        Args:
            key_states:   Shape (Batch, Heads, 1_or_L, d_k)
            value_states: Shape (Batch, Heads, 1_or_L, d_v)
            
        Returns:
            Tuple of concatenated (key_cache, value_cache)
        """
        if self.key_cache is None:
            # First turn / Prefill phase: initialize cache
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            # Decode phase: concatenate new token projections along sequence dimension
            self.key_cache = torch.cat([self.key_cache, key_states], dim=-2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=-2)

        return self.key_cache, self.value_cache

    def clear(self):
        self.key_cache = None
        self.value_cache = None


def cached_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    use_causal_mask: bool = True
) -> torch.Tensor:
    """
    Causal Attention with KV-Cache integration.
    
    Args:
        query: Shape (Batch, Heads, Q_Len, d_k)  --> Q_Len=L in prefill, Q_Len=1 in decode
        key:   Shape (Batch, Heads, Q_Len, d_k)
        value: Shape (Batch, Heads, Q_Len, d_v)
        kv_cache: KVCache instance (optional)
        use_causal_mask: Applies causal lower-triangular mask if True
    """
    # 1. Update and retrieve full Key/Value states from cache if enabled
    if kv_cache is not None:
        key, value = kv_cache.update(key, value)

    # Sequence lengths
    q_len = query.size(-2)      # 1 during decode, L during prefill
    kv_len = key.size(-2)       # Total accumulated history length
    d_k = query.size(-1)

    # 2. Compute Raw Attention Scores: (B, H, Q_Len, d_k) x (B, H, d_k, KV_Len) -> (B, H, Q_Len, KV_Len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # 3. Apply Causal Masking (only needed if processing multiple query tokens at once)
    if use_causal_mask and q_len > 1:
        # Generate triangular causal mask matching (Q_Len, KV_Len)
        mask = torch.tril(torch.ones(q_len, kv_len, device=query.device)).view(1, 1, q_len, kv_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 4. Softmax and Weighted Value Aggregation: (B, H, Q_Len, KV_Len) x (B, H, KV_Len, d_v) -> (B, H, Q_Len, d_v)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output


# =====================================================================
# Verification: Simulate Prefill + Autoregressive Decoding Loop
# =====================================================================
if __name__ == "__main__":
    batch_size, num_heads, d_k = 1, 4, 16
    cache = KVCache()

    print("=== Step 1: Prefill Phase (Prompt of 4 Tokens) ===")
    prompt_len = 4
    q_prompt = torch.randn(batch_size, num_heads, prompt_len, d_k)
    k_prompt = torch.randn(batch_size, num_heads, prompt_len, d_k)
    v_prompt = torch.randn(batch_size, num_heads, prompt_len, d_k)

    out_prefill = cached_scaled_dot_product_attention(
        q_prompt, k_prompt, v_prompt, kv_cache=cache, use_causal_mask=True
    )
    print(f"Prefill Output Shape:      {out_prefill.shape}")
    print(f"Cached Key Matrix Shape:    {cache.key_cache.shape}\n")

    print("=== Step 2: Decode Phase (Token 5 Generation) ===")
    q_step1 = torch.randn(batch_size, num_heads, 1, d_k) # Single new token (Q_Len=1)
    k_step1 = torch.randn(batch_size, num_heads, 1, d_k)
    v_step1 = torch.randn(batch_size, num_heads, 1, d_k)

    out_step1 = cached_scaled_dot_product_attention(
        q_step1, k_step1, v_step1, kv_cache=cache, use_causal_mask=False
    )
    print(f"Token 5 Output Shape:     {out_step1.shape}")
    print(f"Cached Key Matrix Shape:    {cache.key_cache.shape}\n")

    print("=== Step 3: Decode Phase (Token 6 Generation) ===")
    q_step2 = torch.randn(batch_size, num_heads, 1, d_k)
    k_step2 = torch.randn(batch_size, num_heads, 1, d_k)
    v_step2 = torch.randn(batch_size, num_heads, 1, d_k)

    out_step2 = cached_scaled_dot_product_attention(
        q_step2, k_step2, v_step2, kv_cache=cache, use_causal_mask=False
    )
    print(f"Token 6 Output Shape:     {out_step2.shape}")
    print(f"Cached Key Matrix Shape:    {cache.key_cache.shape}")

```

---

## Detailed Code Breakdown

1. **`KVCache` Class:**
* Holds persistent reference to `key_cache` and `value_cache`.
* `torch.cat([self.key_cache, key_states], dim=-2)` dynamically expands the sequence dimension every time a new token's key/value vectors are generated.


2. **Dimension Transformation:**
* During **Prefill** (`prompt_len = 4`): Query is $(1, 4, 4, 16)$, Key/Value are $(1, 4, 4, 16)$, output is $(1, 4, 4, 16)$.
* During **Decode Step 1** (`q_len = 1`): Query is $(1, 4, 1, 16)$, new Key/Value are $(1, 4, 1, 16)$. After concatenation, cached Key/Value become $(1, 4, 5, 16)$. Output shape remains $(1, 4, 1, 16)$.


3. **Causal Mask Optimization:**
* In the decode phase (`q_len = 1`), a single incoming query token can validly attend to **all** previous cached tokens up to the current sequence position. Therefore, causal triangular masking is only computed when `q_len > 1`.



---

---