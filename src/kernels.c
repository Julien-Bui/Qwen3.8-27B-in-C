#include "kernels.h"
#include "quant.h"
#include <immintrin.h>
#include <string.h>
#include <math.h>

/* ============================ AVX2 / FMA ============================
 * Schéma commun : accumulateur vectoriel par ligne, FMA par 8 éléments.
 * _mm256_cvtepi8_epi32 ne convertit que les 8 PREMIERS octets d'un
 * __m128i : chaque vecteur de 16 quants est donc traité en deux moitiés
 * (bytes 0..7, puis bytes 8..15 via _mm_bsrli_si128).
 * Décomposition Q4_K/Q5_K : w = d*q - m  =>  dot = d*Σ(q·x) - m*Σx.
 * ==================================================================== */

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 s = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, s);
    s = _mm_movehl_ps(s, lo);
    lo = _mm_add_ss(lo, s);
    return _mm_cvtss_f32(lo);
}

static inline float hsum256_2(__m256 a, __m256 b) {
    return hsum256(_mm256_add_ps(a, b));
}

/* acc += (q * ds) · x[0..15]  ; q = 16 quants int8, ds = échelle FP32 */
static inline __m256 fmadd16_i8(__m128i q, __m256 ds, const float *xp, __m256 acc) {
    __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(vf, ds), _mm256_loadu_ps(xp), acc);
    vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(q, 8)));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(vf, ds), _mm256_loadu_ps(xp + 8), acc);
    return acc;
}

/* ---------------- F32 ---------------- */
static void dot_f32(const float *w, const float *x, uint64_t n, float *out, uint64_t r) {
    __m256 acc = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc);
    float s = hsum256(acc);
    for (; i < n; i++) s += w[i] * x[i];
    out[r] = s;
}

/* ---------------- Q8_0 ---------------- */
static void dot_q8_0(const block_q8_0 *b, size_t nb, const float *x, float *out, uint64_t r) {
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < nb; i++) {
        __builtin_prefetch(&b[i + 2].d, 0, 3);
        fp16_t dh; memcpy(&dh, &b[i].d, 2);
        const __m256 d = _mm256_set1_ps(fp16_to_f32(dh));
        acc = fmadd16_i8(_mm_loadu_si128((const __m128i *)b[i].qs),
                         d, x + i * 32, acc);
        acc = fmadd16_i8(_mm_loadu_si128((const __m128i *)(b[i].qs + 16)),
                         d, x + i * 32 + 16, acc);
    }
    out[r] = hsum256(acc);
}

/* ---------------- IQ4_XS ---------------- */
static void dot_iq4_xs(const block_iq4_xs *b, size_t nb, const float *x, float *out, uint64_t r) {
    const __m128i lut = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    const __m128i mask4 = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();

    for (size_t i = 0; i < nb; i++) {
        __builtin_prefetch(&b[i + 2].d, 0, 3);
        fp16_t dh; memcpy(&dh, &b[i].d, 2);
        const float d = fp16_to_f32(dh);
        const uint8_t *qs = b[i].qs;

        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((b[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF)
                         | (((b[i].scales_h >> 2 * ib) & 3) << 4);
            const __m256 dl = _mm256_set1_ps(d * (float)(ls - 32));

            const __m128i bytes = _mm_loadu_si128((const __m128i *)(qs + ib * 16));
            const __m128i lo4 = _mm_and_si128(bytes, mask4);                /* elems 0..15  */
            const __m128i hi4 = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask4); /* 16..31 */

            acc = fmadd16_i8(_mm_shuffle_epi8(lut, lo4), dl,
                             x + i * 256 + ib * 32, acc);
            acc = fmadd16_i8(_mm_shuffle_epi8(lut, hi4), dl,
                             x + i * 256 + ib * 32 + 16, acc);
        }
    }
    out[r] = hsum256(acc);
}

/* ------------- Q4_K / Q5_K : tranches de 16 nibbles + échelle const ------------- */
static inline __m128i nib_lo(__m128i bytes, __m128i mask4) {
    return _mm_and_si128(bytes, mask4);
}
static inline __m128i nib_hi(__m128i bytes, __m128i mask4) {
    return _mm_and_si128(_mm_srli_epi16(bytes, 4), mask4);
}

/* Sommes de x par groupes de 32 éléments — partagées par toutes les lignes
 * de la matrice (le terme m*Σx du dot ne dépend pas de la ligne).
 * xs[k] = Σ x[32k .. 32k+31] ; xs doit avoir ne0/32 cases. */
static void xsums32(const float *x, uint64_t n32, float *xs) {
    for (uint64_t k = 0; k < n32; k++) {
        const __m256 a = _mm256_loadu_ps(x + k * 32);
        const __m256 b = _mm256_loadu_ps(x + k * 32 + 8);
        const __m256 c = _mm256_loadu_ps(x + k * 32 + 16);
        const __m256 d = _mm256_loadu_ps(x + k * 32 + 24);
        xs[k] = hsum256(_mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)));
    }
}

/* ------------- Q4_K ------------- */
static void dot_q4_K(const block_q4_K *b, size_t nb, const float *x,
                     const float *xs, float *out, uint64_t r) {
    const __m128i mask4 = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0;

    for (size_t i = 0; i < nb; i++) {
        __builtin_prefetch(&b[i + 2].d, 0, 3);
        fp16_t dh, dmh; memcpy(&dh, &b[i].d, 2); memcpy(&dmh, &b[i].dmin, 2);
        const float d = fp16_to_f32(dh), dm = fp16_to_f32(dmh);

        for (int g = 0; g < 4; g++) {
            uint8_t sc, m;
            get_scale_min_k4(2 * g,     b[i].scales, &sc, &m);
            const __m256 d1 = _mm256_set1_ps(d * sc);
            const float m1 = dm * m;
            get_scale_min_k4(2 * g + 1, b[i].scales, &sc, &m);
            const __m256 d2 = _mm256_set1_ps(d * sc);
            const float m2 = dm * m;

            const uint64_t base = i * 256 + g * 64;
            const __m128i *q16 = (const __m128i *)(b[i].qs + g * 32);
            const __m128i b0 = _mm_loadu_si128(q16);     /* octets 0..15  */
            const __m128i b1 = _mm_loadu_si128(q16 + 1); /* octets 16..31 */

            acc = fmadd16_i8(nib_lo(b0, mask4), d1, x + base,      acc); /* 0..15  */
            acc = fmadd16_i8(nib_lo(b1, mask4), d1, x + base + 16, acc); /* 16..31 */
            acc = fmadd16_i8(nib_hi(b0, mask4), d2, x + base + 32, acc); /* 32..47 */
            acc = fmadd16_i8(nib_hi(b1, mask4), d2, x + base + 48, acc); /* 48..63 */

            accm += m1 * xs[i * 8 + 2 * g] + m2 * xs[i * 8 + 2 * g + 1];
        }
    }
    out[r] = hsum256(acc) - accm;
}

/* ------------- Q5_K ------------- */
static void dot_q5_K(const block_q5_K *b, size_t nb, const float *x,
                     const float *xs, float *out, uint64_t r) {
    const __m128i mask4 = _mm_set1_epi8(15);
    const __m128i one = _mm_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0;

    for (size_t i = 0; i < nb; i++) {
        __builtin_prefetch(&b[i + 2].d, 0, 3);
        fp16_t dh, dmh; memcpy(&dh, &b[i].d, 2); memcpy(&dmh, &b[i].dmin, 2);
        const float d = fp16_to_f32(dh), dm = fp16_to_f32(dmh);

        for (int g = 0; g < 4; g++) {
            uint8_t sc, m;
            get_scale_min_k4(2 * g,     b[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dm * m;
            get_scale_min_k4(2 * g + 1, b[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dm * m;

            const uint64_t base = i * 256 + g * 64;
            const __m128i *q16 = (const __m128i *)(b[i].qs + g * 32);
            /* qh n'est PAS décalé par groupe : l'octet qh[l] porte les 8 bits
             * de flag de l'élément l — bit 2g (moitié basse), bit 2g+1 (haute). */
            const __m128i qha = _mm_loadu_si128((const __m128i *)(b[i].qh));
            const __m128i qhb = _mm_loadu_si128((const __m128i *)(b[i].qh + 16));
            /* srli_epi16 en lanes 16 bits : le bit reste dans l'octet (s <= 7). */
            const __m128i bitla = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qha, 2 * g), one), 4);
            const __m128i bitlb = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qhb, 2 * g), one), 4);
            const __m128i bitha = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qha, 2 * g + 1), one), 4);
            const __m128i bithb = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qhb, 2 * g + 1), one), 4);

            const __m128i b0 = _mm_loadu_si128(q16);
            const __m128i b1 = _mm_loadu_si128(q16 + 1);

            acc = fmadd16_i8(_mm_add_epi8(nib_lo(b0, mask4), bitla),
                             _mm256_set1_ps(d1), x + base, acc);        /* 0..15  */
            acc = fmadd16_i8(_mm_add_epi8(nib_lo(b1, mask4), bitlb),
                             _mm256_set1_ps(d1), x + base + 16, acc);   /* 16..31 */
            acc = fmadd16_i8(_mm_add_epi8(nib_hi(b0, mask4), bitha),
                             _mm256_set1_ps(d2), x + base + 32, acc);   /* 32..47 */
            acc = fmadd16_i8(_mm_add_epi8(nib_hi(b1, mask4), bithb),
                             _mm256_set1_ps(d2), x + base + 48, acc);   /* 48..63 */

            accm += m1 * xs[i * 8 + 2 * g] + m2 * xs[i * 8 + 2 * g + 1];
        }
    }
    out[r] = hsum256(acc) - accm;
}

/* ------------- Q6_K ------------- */
static void dot_q6_K(const block_q6_K *b, size_t nb, const float *x, float *out, uint64_t r) {
    const __m128i mask4 = _mm_set1_epi8(15);
    const __m128i mask3 = _mm_set1_epi8(3);
    const __m128i c32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (size_t i = 0; i < nb; i++) {
        __builtin_prefetch(&b[i + 2].d, 0, 3);
        fp16_t dh; memcpy(&dh, &b[i].d, 2);
        const float d = fp16_to_f32(dh);

        for (int n = 0; n < 2; n++) {
            const __m128i *ql16 = (const __m128i *)(b[i].ql + n * 64);
            const __m128i *qh16 = (const __m128i *)(b[i].qh + n * 32);
            const int8_t *sc = b[i].scales + n * 8;
            const uint64_t base = i * 256 + n * 128;

            for (int hh = 0; hh < 2; hh++) {
                const __m128i qlv  = _mm_loadu_si128(ql16 + hh);     /* ql[l]    */
                const __m128i qlv2 = _mm_loadu_si128(ql16 + hh + 2); /* ql[l+32] */
                const __m128i qhv  = _mm_loadu_si128(qh16 + hh);     /* qh[l]    */
                const uint64_t xb = base + hh * 16;
                __m128i q;

                /* elems l+0 : (ql&0xF)|((qh>>0)&3)<<4, -32 */
                q = _mm_sub_epi8(_mm_add_epi8(
                        _mm_and_si128(qlv, mask4),
                        _mm_slli_epi16(_mm_and_si128(qhv, mask3), 4)), c32);
                acc = fmadd16_i8(q, _mm256_set1_ps(d * sc[hh + 0]), x + xb, acc);

                /* elems l+32 : (ql32&0xF)|((qh>>2)&3)<<4 */
                q = _mm_sub_epi8(_mm_add_epi8(
                        _mm_and_si128(qlv2, mask4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 2), mask3), 4)), c32);
                acc = fmadd16_i8(q, _mm256_set1_ps(d * sc[hh + 2]), x + xb + 32, acc);

                /* elems l+64 : (ql>>4)|((qh>>4)&3)<<4 */
                q = _mm_sub_epi8(_mm_add_epi8(
                        _mm_and_si128(_mm_srli_epi16(qlv, 4), mask4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 4), mask3), 4)), c32);
                acc = fmadd16_i8(q, _mm256_set1_ps(d * sc[hh + 4]), x + xb + 64, acc);

                /* elems l+96 : (ql32>>4)|((qh>>6)&3)<<4 */
                q = _mm_sub_epi8(_mm_add_epi8(
                        _mm_and_si128(_mm_srli_epi16(qlv2, 4), mask4),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 6), mask3), 4)), c32);
                acc = fmadd16_i8(q, _mm256_set1_ps(d * sc[hh + 6]), x + xb + 96, acc);
            }
        }
    }
    out[r] = hsum256(acc);
}

/* ============================ Dispatch ============================ */

/* plus grande ligne = n_ff = 17408 -> 544 groupes de 32 */
#define MAX_XSUM32 544
static _Thread_local float tls_xsum[MAX_XSUM32];

void gemv_rows(const gguf_tensor_info_t *W, const float *x,
               float *out, uint64_t r0, uint64_t r1) {
    const uint64_t ne0 = W->ne[0];
    const size_t be = (W->type == GGML_Q8_0) ? QK8_0 : QK_K;

    if ((W->type == GGML_Q4_K || W->type == GGML_Q5_K) && ne0 / 32 <= MAX_XSUM32)
        xsums32(x, ne0 / 32, tls_xsum);

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row = (const uint8_t *)quant_row_ptr(W, r);
        const uint64_t nb = ne0 / be;
        switch (W->type) {
        case GGML_F32:    dot_f32((const float *)row, x, ne0, out, r);          break;
        case GGML_Q8_0:   dot_q8_0((const block_q8_0 *)row, nb, x, out, r);     break;
        case GGML_IQ4_XS: dot_iq4_xs((const block_iq4_xs *)row, nb, x, out, r); break;
        case GGML_Q4_K:   dot_q4_K((const block_q4_K *)row, nb, x,
                                   (ne0 / 32 <= MAX_XSUM32) ? tls_xsum : NULL, out, r); break;
        case GGML_Q5_K:   dot_q5_K((const block_q5_K *)row, nb, x,
                                   (ne0 / 32 <= MAX_XSUM32) ? tls_xsum : NULL, out, r); break;
        case GGML_Q6_K:   dot_q6_K((const block_q6_K *)row, nb, x, out, r);     break;
        default:          out[r] = 0.0f; break; /* F16 : absent de ce modèle */
        }
    }
}

/* ============================ Parallélisation ============================ */
typedef struct {
    const gguf_tensor_info_t *W;
    const float *x;
    float *out;
} gemv_job_t;

static void gemv_chunk(void *arg, int tid, int ntid) {
    gemv_job_t *j = arg;
    const uint64_t rows = j->W->ne[1];
    gemv_rows(j->W, j->x, j->out, rows * (uint64_t)tid / ntid,
                                   rows * (uint64_t)(tid + 1) / ntid);
}

void gemv(pool_t *pool, const gguf_tensor_info_t *W, const float *x, float *out) {
    gemv_job_t job = { W, x, out };
    if (pool)
        pool_run(pool, gemv_chunk, &job);
    else
        gemv_chunk(&job, 0, 1);
}

/* ======================== Module D1 : Ops ======================== */

/* RMSNorm : out = (x / sqrt(mean(x2) + eps)) * weight */
void rmsnorm(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    __m256 sum256 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum256 = _mm256_fmadd_ps(vx, vx, sum256);
    }
    float sum_sq = hsum256(sum256);
    for (; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float mean_sq = sum_sq / (float)n;
    float scale = 1.0f / sqrtf(mean_sq + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    i = 0;
    if (weight) {
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
            _mm256_storeu_ps(out + i, vo);
        }
        for (; i < n; i++) {
            out[i] = x[i] * scale * weight[i];
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vo = _mm256_mul_ps(vx, vscale);
            _mm256_storeu_ps(out + i, vo);
        }
        for (; i < n; i++) {
            out[i] = x[i] * scale;
        }
    }
}

/* L2Norm : out = x / max(||x||_2, eps) - pas de weight, divise par ||x|| */
void l2norm(float *out, const float *x, uint64_t n, float eps) {
    __m256 sum256 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum256 = _mm256_fmadd_ps(vx, vx, sum256);
    }
    float sum_sq = hsum256(sum256);
    for (; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float norm = sqrtf(sum_sq);
    float denom = (norm > eps) ? norm : eps;
    float scale = 1.0f / denom;
    __m256 vscale = _mm256_set1_ps(scale);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vo = _mm256_mul_ps(vx, vscale);
        _mm256_storeu_ps(out + i, vo);
    }
    for (; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

static inline __m256 exp256_ps(__m256 x) {
    __m256 x_clamp = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)), _mm256_set1_ps(-88.0f));
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x_clamp, _mm256_set1_ps(1.4426950408889634f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.6931471805599453f), x_clamp);
    __m256 p = _mm256_set1_ps(0.000198412698f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.00139304842f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.00833333333f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.04166666666f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.16666666666f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i ik = _mm256_cvtps_epi32(k);
    __m256i pow2 = _mm256_slli_epi32(_mm256_add_epi32(ik, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2));
}

/* silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
void silu_inplace(float *x, uint64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 negx = _mm256_sub_ps(zero, vx);
        __m256 exp_negx = exp256_ps(negx);
        __m256 denom = _mm256_add_ps(one, exp_negx);
        __m256 sig = _mm256_div_ps(one, denom);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, sig));
    }
    for (; i < n; i++) {
        float val = x[i];
        if (val < -20.0f) {
            x[i] = val * expf(val);
        } else if (val > 20.0f) {
            x[i] = val;
        } else {
            x[i] = val / (1.0f + expf(-val));
        }
    }
}

/* sigmoid(x) = 1 / (1 + exp(-x)) */
void sigmoid_inplace(float *x, uint64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 negx = _mm256_sub_ps(zero, vx);
        __m256 exp_negx = exp256_ps(negx);
        __m256 denom = _mm256_add_ps(one, exp_negx);
        _mm256_storeu_ps(x + i, _mm256_div_ps(one, denom));
    }
    for (; i < n; i++) {
        float val = x[i];
        if (val < -20.0f) {
            x[i] = 0.0f;
        } else if (val > 20.0f) {
            x[i] = 1.0f;
        } else {
            x[i] = 1.0f / (1.0f + expf(-val));
        }
    }
}

/* softplus(x) = log(1 + exp(x)) robuste au debordement exp */
void softplus_inplace(float *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        float val = x[i];
        if (val > 30.0f) {
            x[i] = val;
        } else if (val < -30.0f) {
            x[i] = expf(val);
        } else {
            x[i] = log1pf(expf(val));
        }
    }
}

/* softmax : soustraire le max d'abord ! */
void softmax_inplace(float *x, uint64_t n) {
    if (n == 0) return;
    float max_val = x[0];
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum_exp = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        float e = expf(x[i] - max_val);
        x[i] = e;
        sum_exp += (double)e;
    }
    float inv_sum = (float)(1.0 / sum_exp);
    uint64_t i = 0;
    __m256 vinv = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, vinv));
    }
    for (; i < n; i++) {
        x[i] *= inv_sum;
    }
}
