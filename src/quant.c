#include "quant.h"
#include <string.h>
#include <math.h>

/* ---------- FP16 -> FP32 portable (F16C viendra avec les kernels AVX2) ---------- */
float fp16_to_f32(fp16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                          /* +-0 */
        } else {                                  /* dénormalisé : renormaliser */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (man << 13);   /* inf / NaN */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

size_t quant_block_size(ggml_type_t t) {
    switch (t) {
    case GGML_F32:    return 4;
    case GGML_F16:    return 2;
    case GGML_Q8_0:   return sizeof(block_q8_0);
    case GGML_Q4_K:   return sizeof(block_q4_K);
    case GGML_Q5_K:   return sizeof(block_q5_K);
    case GGML_Q6_K:   return sizeof(block_q6_K);
    case GGML_IQ4_XS: return sizeof(block_iq4_xs);
    default:          return 0;
    }
}

/* Éléments couverts par un bloc selon le format. */
static size_t block_elems(ggml_type_t t) {
    switch (t) {
    case GGML_F32:  return 1;
    case GGML_F16:  return 1;
    case GGML_Q8_0: return QK8_0;
    case GGML_Q4_K:
    case GGML_Q5_K:
    case GGML_Q6_K:
    case GGML_IQ4_XS: return QK_K;
    default:        return 0;
    }
}

size_t quant_row_bytes(ggml_type_t t, uint64_t ne0) {
    size_t be = block_elems(t);
    if (be == 0) return 0;
    return (size_t)(ne0 / be) * quant_block_size(t);
}

const void *quant_row_ptr(const gguf_tensor_info_t *t, uint64_t row) {
    return (const uint8_t *)t->data + (size_t)row * quant_row_bytes(t->type, t->ne[0]);
}

const char *quant_type_name(ggml_type_t t) {
    switch (t) {
    case GGML_F32:    return "F32";
    case GGML_F16:    return "F16";
    case GGML_Q8_0:   return "Q8_0";
    case GGML_Q4_K:   return "Q4_K";
    case GGML_Q5_K:   return "Q5_K";
    case GGML_Q6_K:   return "Q6_K";
    case GGML_IQ4_XS: return "IQ4_XS";
    default:          return "UNSUPPORTED";
    }
}

/* ---------- Déquantification de référence (portées de ggml-quants.c) ---------- */

static void dequant_q8_0(const block_q8_0 *x, float *y, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        const float d = fp16_to_f32(x[i].d);
        for (int l = 0; l < QK8_0; l++)
            y[i * QK8_0 + l] = d * x[i].qs[l];
    }
}

static void dequant_q4_K(const block_q4_K *x, float *y, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = fp16_to_f32(x[i].d);
        const float min = fp16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void dequant_q5_K(const block_q5_K *x, float *y, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d   = fp16_to_f32(x[i].d);
        const float min = fp16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++)
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void dequant_q6_K(const block_q6_K *x, float *y, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        const float d = fp16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void dequant_iq4_xs(const block_iq4_xs *x, float *y, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        const uint8_t *qs = x[i].qs;
        const float d = fp16_to_f32(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ib++) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF)
                         | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                y[j]      = dl * kvalues_iq4nl[qs[j] & 0xF];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

int dequant_row(ggml_type_t t, const void *data, uint64_t ne0, float *dst) {
    const size_t be = block_elems(t);
    if (be == 0) return -1;
    if (ne0 % be != 0) return -1;
    const uint64_t n_blocks = ne0 / be;

    switch (t) {
    case GGML_F32:
        memcpy(dst, data, (size_t)ne0 * 4);
        break;
    case GGML_F16: {
        const fp16_t *h = data;
        for (uint64_t i = 0; i < ne0; i++)
            dst[i] = fp16_to_f32(h[i]);
        break;
    }
    case GGML_Q8_0:   dequant_q8_0(data, dst, n_blocks);   break;
    case GGML_Q4_K:   dequant_q4_K(data, dst, n_blocks);   break;
    case GGML_Q5_K:   dequant_q5_K(data, dst, n_blocks);   break;
    case GGML_Q6_K:   dequant_q6_K(data, dst, n_blocks);   break;
    case GGML_IQ4_XS: dequant_iq4_xs(data, dst, n_blocks); break;
    default:          return -1;
    }
    return 0;
}
