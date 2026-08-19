# Qwen 3.8-27B Inference Engine in Pure C

> **A self-contained, CPU-native C inference engine (C11 + AVX2 + FMA) paired with a lightweight Python SSE streaming server and a minimalist web interface for the hybrid Qwen 3.8-27B architecture.**

---

## Overview

This project provides an end-to-end local LLM inference stack designed to run the **Qwen3.8-27B-IQ4_XS.gguf** model (14.70 GB quantized weights) on standard x86-64 CPUs with **zero heavy frameworks** (no PyTorch, no CUDA, no ONNX, no pip dependencies):

1. **Pure C Inference Engine (`build/qwen`)**: High-performance 64-layer forward pass with hand-tuned AVX2/FMA kernels, native MTP speculative decoding, and a 248k BPE tokenizer.
2. **Python Streaming Server (`server.py`)**: Zero-dependency standard-library HTTP server that streams generated tokens in real time via Server-Sent Events (SSE).
3. **Minimalist Web Interface (`web/`)**: Distraction-free, single-canvas chat interface inspired by Anthropic (Claude.ai).

```
+-------------------------------------------------------------------+
|                    Minimalist Web Interface                       |
|           (Vanilla JS + CSS, Single Canvas, SSE Client)           |
+-------------------------------------------------------------------+
                                  ^  (HTTP / Server-Sent Events)
                                  v
+-------------------------------------------------------------------+
|                    Python Streaming Server                        |
|            (server.py: Standard library http.server)              |
+-------------------------------------------------------------------+
                                  ^  (Subprocess stdout stream)
                                  v
+-------------------------------------------------------------------+
|                   Pure C11 Inference Engine                       |
|       (build/qwen: AVX2 Kernels + MTP Speculative Decoding)       |
+-------------------------------------------------------------------+
                                  |
                                  v
                    Qwen3.8-27B-IQ4_XS.gguf (14.7 GB)
```

---

## Key Technical Features

### Pure C11 Inference Core
- **Zero External C Libraries**: 100% standard C11 relying only on `libc`, `libm`, and POSIX `pthread`.
- **64-Layer Hybrid Backbone Architecture**:
  - **48 Gated DeltaNet Layers**: State-Space Model (SSM) featuring causal linear recurrence $S_h \in \mathbb{R}^{128 \times 128}$, 4-step causal Conv1D, per-head exponential decay, and L2Norm.
  - **16 Full Attention Layers**: Grouped Query Attention (GQA 24/4), partial RoPE with section restart $\{11, 11, 10, 0\}$, Q/K RMSNorm, and sigmoid gating.
- **Speculative Decoding with Native MTP Layer**:
  - Uses the native Multi-Token Prediction layer (`blk.64.*`) to draft future tokens at ~1/64th of the backbone cost.
  - **Batched GEMV (`gemv_batch`)**: Verifies multiple draft candidate tokens in a single memory sweep over the 14.70 GB weights.
  - **Instant State Rollback**: Saves and restores GDN Conv1D and SSM recurrent states in microseconds (`model_rollback_to`), preserving 100% strict mathematical equivalence with standard greedy decoding.
- **Hand-Tuned AVX2 + FMA Kernels**:
  - SIMD GEMV for `IQ4_XS`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `FP32`.
  - 4-accumulator unrolling to saturate dual FMA ports.
  - Polynomial vector approximations for `silu_inplace` and `sigmoid_inplace` (`exp256_ps`).
- **248k GPT-2 Byte-Level BPE Tokenizer**:
  - Complete byte mapping (`0x20` $\to$ `Ġ`, `0x0A` $\to$ `Ċ`), hash table merges lookup, and lossless UTF-8 string decoding without mojibake.
- **Stack Min-Heap Sampler**:
  - Top-K candidate extraction in $O(V \log K)$ using a fixed stack-allocated min-heap (0.36 ms vs 30 ms previously), Nucleus Top-P, Temperature scaling, Repetition Penalty, and XorShift64 PRNG.

### Python Streaming Backend & Web UI
- **Zero-Dependency Python Backend (`server.py`)**: Uses only the Python 3 standard library (`http.server`, `subprocess`, `threading`, `json`, `urllib.parse`). No `pip install` required.
- **Real-Time Token Streaming**: Streams tokens from the C binary directly to the browser using Server-Sent Events (`text/event-stream`).
- **Process Lifecycle Management**: Thread-safe subprocess handling with immediate cancellation support via `/api/stop`.
- **Minimalist Web UI (`web/`)**:
  - Distraction-free single-canvas layout focused entirely on reading and writing.
  - Live markdown formatting with code syntax blocks and 1-click clipboard copy.
  - Auto-expanding prompt input with keyboard shortcuts (`Enter` to submit, `Shift + Enter` for newlines).
  - Single-click screen clear button (`New Chat`).

---

## Project Structure

```
.
├── Makefile                 # Build system configuration and test targets
├── Qwen3.8-27B-IQ4_XS.gguf  # Quantized GGUF model weights (14.70 GB)
├── server.py                # Zero-dependency Python HTTP & SSE streaming server
├── web/                     # Minimalist single-canvas web interface
│   ├── index.html           # Structure and layout
│   ├── style.css            # Neutral dark minimalist styling
│   └── app.js               # EventSource streaming & markdown parser
├── include/
│   ├── gguf.h              # GGUF v3 binary parser and tensor loader
│   ├── quant.h             # Quantization block structures & dequantization API
│   ├── kernels.h           # AVX2 kernels (GEMV, Batched GEMV, RMSNorm, L2Norm, SiLU)
│   ├── threadpool.h        # POSIX thread pool with barrier synchronization
│   ├── model.h             # Architecture structs, KV cache, and MTP layer state
│   ├── tokenizer.h         # 248k BPE tokenizer (encoder / decoder)
│   └── sampler.h           # Top-K / Top-P / Temperature sampler
└── src/
    ├── gguf.c              # Zero-copy mmap loading of 963 tensors
    ├── quant.c             # Scalar reference dequantization
    ├── kernels.c           # Hand-tuned AVX2 + FMA SIMD kernels
    ├── threadpool.c        # Work-stealing thread management and GEMV partitioning
    ├── model.c             # 64-layer forward pass & MTP speculative decoding
    ├── tokenizer.c         # BPE encoding, byte fallback, and decoding
    ├── sampler.c           # Min-heap Top-K selection and nucleus sampling
    ├── main.c              # Interactive CLI generation tool
    ├── test_ops.c          # Mathematical unit test suite (37,016 assertions)
    ├── test_tokenizer.c    # Tokenizer multilingual & code validation
    ├── test_forward.c      # 64-layer forward pass test harness
    ├── test_batch.c        # Batched GEMV mathematical equivalence test
    ├── test_forward_batch.c# Batched forward pass validation
    └── test_kernels.c      # AVX2 memory bandwidth benchmark
```

---

## Build & Usage Guide

### 1. Prerequisites
- **C Compiler**: GCC (with C11 support)
- **CPU**: x86-64 CPU supporting **AVX2** and **FMA** (Intel Haswell+ or AMD Zen2+)
- **Python**: Python 3.8+ (Standard library only, no packages to install)
- **Model**: `Qwen3.8-27B-IQ4_XS.gguf` placed at the project root

### 2. Compilation
```bash
make -j8
```

### 3. Running the Web Interface (Recommended)
```bash
make serve
```
or directly with Python:
```bash
python3 server.py --port 8080
```
Then open **`http://localhost:8080`** in your browser.

### 4. Running Text Generation from CLI

**Fast Speculative Decoding Mode (Recommended):**
```bash
./build/qwen -m Qwen3.8-27B-IQ4_XS.gguf -p "Hello! Introduce yourself in one sentence:" -n 32 --threads 12 --spec 1 --temp 0
```

**Standard Sampling Mode:**
```bash
./build/qwen -m Qwen3.8-27B-IQ4_XS.gguf -p "Write a C function to reverse a string:" -n 48 --threads 12 --temp 0.7
```

### 5. CLI Arguments Reference
| Argument | Description | Default |
|---|---|---|
| `-m, --model` | Path to GGUF model file | `Qwen3.8-27B-IQ4_XS.gguf` |
| `-p, --prompt` | Input prompt text | `"Hello! Introduce yourself in one sentence:"` |
| `-n, --n-predict` | Number of tokens to generate | `32` |
| `-t, --threads` | Number of CPU worker threads | `12` |
| `-c, --ctx-size` | Allocated context length | `512` |
| `--spec` | MTP speculative drafts count (`0` = off, `1` = on) | `0` |
| `--temp` | Sampling temperature (`0.0` = greedy) | `0.70` |
| `--top-p` | Nucleus sampling cutoff (Top-P) | `0.90` |
| `--top-k` | Top-K candidate pool size | `40` |
| `--repeat-penalty` | Repetition penalty factor | `1.10` |

---

## Server API Endpoints

The Python server (`server.py`) exposes the following endpoints:

### `POST /api/chat`
Starts a streaming inference session using Server-Sent Events (SSE).

**Request Body (JSON):**
```json
{
  "prompt": "Explain AVX2 SIMD in two sentences.",
  "temp": 0.0,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "n_predict": 128,
  "threads": 12
}
```

**Response:**
Stream of Server-Sent Events:
```
data: {"type": "token", "content": "AVX2"}

data: {"type": "token", "content": " expands"}

data: {"type": "done"}
```

### `POST /api/stop`
Terminates the active inference process immediately.

---

## Test Suite and Verification

| Command | Objective | Result |
|---|---|---|
| `./build/test_ops` | Mathematical validation of ops (RMSNorm, L2Norm, SiLU, Softmax) | **37,016 assertions passed (0 failures)** |
| `./build/test_tokenizer Qwen3.8-27B-IQ4_XS.gguf` | Validation of 248k BPE Tokenizer (French, English, Code, Math) | **100% Match (zero mojibake)** |
| `./build/test_batch Qwen3.8-27B-IQ4_XS.gguf` | Verification of batched GEMV against single-token GEMV | **Max diff = 0.000e+00** |
| `./build/test_forward_batch Qwen3.8-27B-IQ4_XS.gguf 3` | Batched forward pass equivalence check | **Max diff = 0.000e+00** |
| `./build/test_kernels Qwen3.8-27B-IQ4_XS.gguf 12` | AVX2 GEMV memory bandwidth benchmark | **18.55 GB/s sustained throughput** |

---

## Performance and Benchmarks

Measured on Intel Core i9-12900H (12 threads, DDR5 RAM under WSL2):

| Component | Baseline | Optimized | Speedup |
|---|---|---|---|
| **Top-K Sampler** | 30.0 ms (full 248k `qsort`) | **0.36 ms** (Stack Min-Heap) | **80x faster** |
| **SiLU Activation** | ~10.0 ms (scalar `expf`) | **1.14 ms** (AVX2 `exp256_ps`) | **8.7x faster** |
| **Prefill Speed** | 0.74 tok/s | **1.25 tok/s** | **+68.9 %** |
| **Generation (Standard Greedy)** | 1.06 tok/s | **1.23 tok/s** | **+16.0 %** |
| **Generation (Speculative MTP `--spec 1`)** | 1.06 tok/s | **1.64 tok/s** (80-90% draft acceptance) | **+54.7 %** |

---

## Recommended WSL2 Configuration

To prevent OS swap thrashing with the 14.70 GB model, allocate 28 GB of RAM to WSL2 in `C:\Users\<username>\.wslconfig`:

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
