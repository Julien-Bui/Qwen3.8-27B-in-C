# ⚡ Qwen 3.8-27B Inference Engine in Pure C

> **A 100% CPU-native, pure C inference engine (C11 + AVX2 + FMA) designed for the hybrid Qwen 3.8-27B architecture (48 Gated DeltaNet + 16 Full Attention layers).**

---

## 📌 Overview

This project provides a complete, self-contained, and highly optimized inference engine to run the **Qwen3.8-27B-IQ4_XS.gguf** model (14.70 GB quantized weights) directly on CPU with **zero external dependencies** (no PyTorch, no CUDA, no ONNX, no third-party libraries).

### ⚙️ Key Technical Features

- **Zero External Frameworks**: 100% pure C11 (depends only on standard `libc`, `libm`, and `pthread`).
- **64-Layer Hybrid Architecture**:
  - **48 Gated DeltaNet Layers**: State-Space Model (SSM) featuring causal matrix linear recurrence $S_h \in \mathbb{R}^{128 \times 128}$, 4-step causal Conv1D, per-head exponential decay, and L2Norm.
  - **16 Full Attention Layers**: Grouped Query Attention (GQA 24/4), partial RoPE with section restart $\{11, 11, 10, 0\}$, Q/K RMSNorm, and sigmoid gating.
- **AVX2 + FMA Quantized Kernels**: Hand-tuned SIMD GEMV kernels for IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0, and FP32 with 4-accumulator unrolling to saturate dual FMA execution ports.
- **248k BPE Tokenizer**: Full GPT-2 byte-level BPE mapping (handling spaces `0x20` $\to$ `Ġ` and newlines `0x0A` $\to$ `Ċ`), hash table lookups, and lossless UTF-8 reconstruction without mojibake.
- **Stack-Allocated Min-Heap Sampler**: $O(V \log K)$ Top-K selection using a fixed stack-allocated min-heap (0.36 ms vs 30 ms previously), Nucleus Top-P, Temperature scaling, Repetition Penalty, and XorShift64 PRNG.
- **Hybrid Topology Threadpool**: Dynamic work distribution tuned for Intel Alder Lake hybrid architectures (6 P-cores + 8 E-cores).

---

## 🗺️ Project Structure

```
.
├── Makefile                 # Build system and test targets
├── Qwen3.8-27B-IQ4_XS.gguf  # GGUF model weights (14.70 GB)
├── include/
│   ├── gguf.h              # GGUF v3 binary parser and tensor loader
│   ├── quant.h             # Quantization block types & dequantization API
│   ├── kernels.h           # AVX2 kernels (GEMV, RMSNorm, L2Norm, SiLU, Sigmoid)
│   ├── threadpool.h        # POSIX thread pool with barrier synchronization
│   ├── model.h             # Hybrid model architecture, KV cache & GDN state
│   ├── tokenizer.h         # 248k BPE tokenizer (encoder / decoder)
│   └── sampler.h           # Top-K / Top-P / Temperature / Repetition penalty
└── src/
    ├── gguf.c              # Zero-copy mmap loading and parsing of 963 tensors
    ├── quant.c             # Scalar reference dequantization
    ├── kernels.c           # Vectorized AVX2 + FMA kernels
    ├── threadpool.c        # Multithreading management and GEMV partitioning
    ├── model.c             # 64-layer forward pass (DeltaNet + GQA Attention)
    ├── tokenizer.c         # BPE encoding, byte fallback, and decoding
    ├── sampler.c           # Min-heap Top-K selection and nucleus sampling
    ├── main.c              # Interactive CLI generation tool
    ├── test_ops.c          # 37,016 mathematical unit tests (Module D1)
    ├── test_tokenizer.c    # Tokenizer validation across multilingual & code inputs
    ├── test_forward.c      # 64-layer forward pass validation & logit verification
    └── test_kernels.c      # AVX2 memory bandwidth benchmark
```

---

## 🚀 Build & Usage Guide

### 1. Prerequisites

- GCC with C11 support
- x86-64 CPU supporting **AVX2** and **FMA** (Intel Haswell+ or AMD Zen2+)
- The model file `Qwen3.8-27B-IQ4_XS.gguf` placed at the project root

### 2. Compilation

```bash
make -j8
```

### 3. Running Text Generation

**General Conversation:**

```bash
./build/qwen -m Qwen3.8-27B-IQ4_XS.gguf -p "Hello! Introduce yourself in two sentences:" -n 32 --threads 12
```

**Deterministic Code Generation (`--temp 0`):**

```bash
./build/qwen -m Qwen3.8-27B-IQ4_XS.gguf -p "Write a C function to reverse a string in place:" -n 48 --threads 12 --temp 0
```

### 4. CLI Arguments Reference

| Argument           | Description                         | Default Value             |
| ------------------ | ----------------------------------- | ------------------------- |
| `-m, --model`      | Path to GGUF model file             | `Qwen3.8-27B-IQ4_XS.gguf` |
| `-p, --prompt`     | Input prompt text                   | `"Bonjour !"`             |
| `-n, --n-predict`  | Number of tokens to generate        | `32`                      |
| `-t, --threads`    | Number of CPU threads               | `12`                      |
| `--temp`           | Sampling temperature (0.0 = greedy) | `0.70`                    |
| `--top-p`          | Nucleus sampling cutoff (Top-P)     | `0.90`                    |
| `--top-k`          | Top-K candidate pool size           | `40`                      |
| `--repeat-penalty` | Repetition penalty factor           | `1.10`                    |

---

## 🧪 Test Suite & Verification

| Target Command                                    | Objective                                                            | Result                                    |
| ------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------- |
| `./build/test_ops`                                | Mathematical verification of D1 ops (RMSNorm, L2Norm, SiLU, Softmax) | **37,016 assertions passed (0 failures)** |
| `./build/test_tokenizer Qwen3.8-27B-IQ4_XS.gguf`  | Validation of 248k BPE Tokenizer (French, English, Code, Math)       | **100% Match (zero mojibake)**            |
| `./build/test_forward Qwen3.8-27B-IQ4_XS.gguf`    | Full 64-layer forward pass & logit distribution check                | **Valid logits & greedy token confirmed** |
| `./build/test_kernels Qwen3.8-27B-IQ4_XS.gguf 12` | AVX2 GEMV memory bandwidth benchmark                                 | **18.55 GB/s sustained throughput**       |

---

## ⚡ Performance Optimizations

| Component            | Before Optimization         | After Optimization               | Speedup            |
| -------------------- | --------------------------- | -------------------------------- | ------------------ |
| **Top-K Sampler**    | 30.0 ms (full 248k `qsort`) | **0.36 ms** (Stack Min-Heap)     | **80x faster** 🚀  |
| **SiLU Activation**  | ~10.0 ms (scalar `expf`)    | **1.14 ms** (AVX2 + `exp256_ps`) | **8.7x faster** 🚀 |
| **Prefill Speed**    | 0.74 tok/s (2.69 s)         | **1.07 tok/s** (1.86 s)          | **+44.6 %**        |
| **Generation Speed** | 1.06 tok/s (15.15 s)        | **1.23 tok/s** (12.98 s)         | **+16.0 %**        |

---

## 💡 Recommended WSL2 Configuration

To prevent OS swap thrashing (the model weights take 14.70 GB of RAM), allocate 28 GB of RAM to WSL2 in `C:\Users\<your_username>\.wslconfig`:

```ini
[wsl2]
memory=28GB
processors=14
swap=8GB
```

Then restart WSL from Windows PowerShell:

```powershell
wsl --shutdown
```
