#ifndef QUANT_H
#define QUANT_H

#include "gguf.h"
#include <stdint.h>
#include <stddef.h>

#define QK_K 256   /* Elements per K-quant super-block */
#define QK8_0 32

typedef uint16_t fp16_t;
float fp16_to_f32(fp16_t h);

/* --- Q8_0 : fp16 d + 32 x int8 (34 bytes) --- */
typedef struct {
    fp16_t d;
    int8_t qs[QK8_0];
} block_q8_0;

/* 6-bit scales & mins shared by Q4_K / Q5_K */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((uint8_t)(q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((uint8_t)(q[j] >> 6) << 4);
    }
}

/* --- Q4_K : 144 bytes / 256 elements --- */
typedef struct {
    fp16_t  d;
    fp16_t  dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K;

/* --- Q5_K : 176 bytes / 256 elements --- */
typedef struct {
    fp16_t  d;
    fp16_t  dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

/* --- Q6_K : 210 bytes / 256 elements --- */
typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    fp16_t  d;
} block_q6_K;

/* --- IQ4_XS : 136 bytes / 256 elements --- */
typedef struct {
    fp16_t   d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
} block_iq4_xs;

_Static_assert(sizeof(block_q8_0) == 34,  "q8_0 size mismatch");
_Static_assert(sizeof(block_q4_K) == 144, "q4_K size mismatch");
_Static_assert(sizeof(block_q5_K) == 176, "q5_K size mismatch");
_Static_assert(sizeof(block_q6_K) == 210, "q6_K size mismatch");
_Static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs size mismatch");

/* Non-linear lookup table for IQ4 format */
extern const int8_t kvalues_iq4nl[16];

/* Block size in bytes (0 if unsupported / variable) */
size_t quant_block_size(ggml_type_t t);

/* Byte size of a row with ne0 elements */
size_t quant_row_bytes(ggml_type_t t, uint64_t ne0);

/* Pointer to row of tensor t */
const void *quant_row_ptr(const gguf_tensor_info_t *t, uint64_t row);

/* Dequantizes ne0 elements of a row (scalar reference) */
int dequant_row(ggml_type_t t, const void *data, uint64_t ne0, float *dst);

const char *quant_type_name(ggml_type_t t);

#endif /* QUANT_H */
