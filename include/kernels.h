#ifndef KERNELS_H
#define KERNELS_H

#include "gguf.h"
#include "threadpool.h"

/* Quantized GEMV: out[r] = dot(x, row r of W) for r in [r0, r1).
 * W is 2-D [ne0 = input_dim, ne1 = output_dim], x has ne0 elements.
 * Direct dispatch by format: F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS.
 * On-the-fly streaming: each weight byte is read exactly once (memory bandwidth bound). */
void gemv_rows(const gguf_tensor_info_t *W, const float *x,
               float *out, uint64_t r0, uint64_t r1);

/* Multithreaded GEMV: partitions rows across thread pool. */
void gemv(pool_t *pool, const gguf_tensor_info_t *W, const float *x, float *out);

/* Batched GEMV (for speculative decoding): out[t][r] = dot(x[t], row r)
 * for n <= 4 vectors, x : [n][ldx], out : [n][ldo]. Weights read only once. */
void gemv_rows_batch(const gguf_tensor_info_t *W, const float *x, uint64_t ldx, int n,
                     float *out, uint64_t ldo, uint64_t r0, uint64_t r1);
void gemv_batch(pool_t *pool, const gguf_tensor_info_t *W,
                const float *x, uint64_t ldx, int n,
                float *out, uint64_t ldo);

/* Ops */
void rmsnorm(float *out, const float *x, const float *weight, uint64_t n, float eps);
void l2norm(float *out, const float *x, uint64_t n, float eps);

/* In-place vectorized activations */
void silu_inplace(float *x, uint64_t n);       /* x * sigmoid(x) */
void sigmoid_inplace(float *x, uint64_t n);    /* 1 / (1 + exp(-x)) */
void softplus_inplace(float *x, uint64_t n);  /* log(1 + exp(x)) */
void softmax_inplace(float *x, uint64_t n);    /* exp(x - max) / sum(exp(x - max)) */

#endif /* KERNELS_H */
