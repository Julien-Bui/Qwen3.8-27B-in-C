#ifndef KERNELS_H
#define KERNELS_H

#include "gguf.h"
#include "threadpool.h"

/* GEMV quantifiee : out[r] = dot(x, ligne r de W) pour r dans [ro, r1).
 * W est 2-D [ne0 = entree, ne1 = sorties], x a ne0 elements.
 * Dispatch par format : F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS.
 * Dequantification a la volee : chaque octet de poids est lu exactement
 * une fois (le kernel est concu pour etre borne par la bande passante). */
void gemv_rows(const gguf_tensor_info_t *W, const float *x,
               float *out, uint64_t r0, uint64_t r1);

/* Version parallele : decoupe les lignes sur le pool de threads. */
void gemv(pool_t *pool, const gguf_tensor_info_t *W, const float *x, float *out);

/* ======================== Module D1 : Ops ======================== */
/* RMSNorm : out = (x / sqrt(mean(x2) + eps)) * weight */
void rmsnorm(float *out, const float *x, const float *weight, uint64_t n, float eps);

/* L2Norm : out = x / max(V|xV_2, eps) - pas de weight, divise par VxVW */
void l2norm(float *out, const float *x, uint64_t n, float eps);

/* Activations in-place */
void silu_inplace(float *x, uint64_t n);       /* x * sigmoid(x) */
void sigmoid_inplace(float *x, uint64_t n);    /* 1 / (1 + exp(-x)) */
void softplus_inplace(float *x, uint64_t n);  /* log(1 + exp(x)) robuste au debordement */
void softmax_inplace(float *x, uint64_t n);    /* exp(x - max) / sum(exp(x - max)) */

#endif
