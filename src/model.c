#define _POSIX_C_SOURCE 200809L

#include "model.h"
#include "kernels.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

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

static void *alloc64(size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, 64, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

static const gguf_tensor_info_t *must_find(const gguf_context_t *g, const char *name) {
    const gguf_tensor_info_t *t = gguf_find_tensor(g, name);
    if (!t) fprintf(stderr, "model_init: missing tensor %s\n", name);
    return t;
}

static int check2(const gguf_tensor_info_t *t, uint64_t ne0, uint64_t ne1, const char *what) {
    if (!t) return -1;
    if (t->n_dims != 2 || t->ne[0] != ne0 || t->ne[1] != ne1) {
        fprintf(stderr, "model_init: %s dimension mismatch\n", what);
        return -1;
    }
    return 0;
}

static int check1(const gguf_tensor_info_t *t, uint64_t ne0, const char *what) {
    if (!t) return -1;
    if (t->n_dims != 1 || t->ne[0] != ne0) {
        fprintf(stderr, "model_init: %s dimension mismatch\n", what);
        return -1;
    }
    return 0;
}

int model_init(qwen_model_t *m, const char *path, uint32_t n_ctx) {
    memset(m, 0, sizeof *m);
    if (gguf_open(&m->gguf, path) != 0) return -1;
    m->cfg = m->gguf.config;
    qwen_dims_t *d = &m->dims;
    qwen_config_t *c = &m->cfg;
    d->n_embd = c->n_embd;
    d->n_ff = c->n_ff;
    d->n_head = c->n_head;
    d->n_kv_head = c->n_head_kv;
    d->head_dim = c->head_dim;
    d->n_k_heads = c->ssm_group_count;
    d->head_k_dim = c->ssm_state_size;
    d->head_v_dim = c->ssm_state_size;
    d->d_inner = c->ssm_inner_size;
    d->d_conv = c->ssm_conv_kernel;
    d->d_state = c->ssm_state_size;
    memcpy(d->rope_sections, c->rope_sections, sizeof d->rope_sections);

    char name[64];
    snprintf(name, sizeof name, "blk.0.ssm_a");
    const gguf_tensor_info_t *a0 = gguf_find_tensor(&m->gguf, name);
    if (!a0 || a0->n_dims != 1) goto fail;
    d->n_v_heads = (uint32_t)a0->ne[0];
    d->conv_dim = 2 * d->n_k_heads * d->head_k_dim + d->d_inner;
    d->n_layer = (c->n_layer > c->nextn_layers) ? c->n_layer - c->nextn_layers : c->n_layer;
    if (d->d_inner != d->n_v_heads * d->head_v_dim) goto fail;

    m->tok_embd = must_find(&m->gguf, "token_embd.weight");
    m->output_norm = must_find(&m->gguf, "output_norm.weight");
    m->output = gguf_find_tensor(&m->gguf, "output.weight");
    if (!m->output) m->output = m->tok_embd;
    if (!m->tok_embd || !m->output_norm) goto fail;
    if (check2(m->tok_embd, d->n_embd, c->vocab_size, "token_embd")) goto fail;

    m->layers = calloc(d->n_layer, sizeof *m->layers);
    if (!m->layers) goto fail;
    uint32_t n_attn = 0, n_recr = 0;
    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        L->is_recurrent = ((il + 1) % c->full_attn_interval) != 0;
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", il);
        if (!(L->attn_norm = must_find(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.post_attention_norm.weight", il);
        if (!(L->post_attn_norm = must_find(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.ffn_gate.weight", il);
        if (!(L->ffn_gate = must_find(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", il);
        if (!(L->ffn_up = must_find(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", il);
        if (!(L->ffn_down = must_find(&m->gguf, name))) goto fail;
        if (check2(L->ffn_gate, d->n_embd, d->n_ff, "ffn_gate") || check2(L->ffn_up, d->n_embd, d->n_ff, "ffn_up") || check2(L->ffn_down, d->n_ff, d->n_embd, "ffn_down")) goto fail;

        if (L->is_recurrent) {
            snprintf(name, sizeof name, "blk.%u.attn_qkv.weight", il);
            if (!(L->wqkv = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.attn_gate.weight", il);
            if (!(L->wqkv_gate = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_conv1d.weight", il);
            if (!(L->ssm_conv1d = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_dt.bias", il);
            if (!(L->ssm_dt_bias = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_a", il);
            if (!(L->ssm_a = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_beta.weight", il);
            if (!(L->ssm_beta = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_alpha.weight", il);
            if (!(L->ssm_alpha = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_norm.weight", il);
            if (!(L->ssm_norm = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.ssm_out.weight", il);
            if (!(L->ssm_out = must_find(&m->gguf, name))) goto fail;
            if (check2(L->wqkv, d->n_embd, d->conv_dim, "wqkv") || check2(L->wqkv_gate, d->n_embd, d->d_inner, "wqkv_gate") || check2(L->ssm_conv1d, d->d_conv, d->conv_dim, "ssm_conv1d") || check2(L->ssm_beta, d->n_embd, d->n_v_heads, "ssm_beta") || check2(L->ssm_alpha, d->n_embd, d->n_v_heads, "ssm_alpha") || check2(L->ssm_out, d->d_inner, d->n_embd, "ssm_out") || check1(L->ssm_dt_bias, d->n_v_heads, "ssm_dt_bias") || check1(L->ssm_a, d->n_v_heads, "ssm_a") || check1(L->ssm_norm, d->head_v_dim, "ssm_norm")) goto fail;
            n_recr++;
        } else {
            snprintf(name, sizeof name, "blk.%u.attn_q.weight", il);
            if (!(L->attn_q = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.attn_k.weight", il);
            if (!(L->attn_k = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.attn_v.weight", il);
            if (!(L->attn_v = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.attn_q_norm.weight", il);
            if (!(L->attn_q_norm = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.attn_k_norm.weight", il);
            if (!(L->attn_k_norm = must_find(&m->gguf, name))) goto fail;
            snprintf(name, sizeof name, "blk.%u.attn_output.weight", il);
            if (!(L->attn_out = must_find(&m->gguf, name))) goto fail;
            if (check2(L->attn_q, d->n_embd, 2 * d->n_head * d->head_dim, "attn_q") || check2(L->attn_k, d->n_embd, d->n_kv_head * d->head_dim, "attn_k") || check2(L->attn_v, d->n_embd, d->n_kv_head * d->head_dim, "attn_v") || check2(L->attn_out, d->n_head * d->head_dim, d->n_embd, "attn_out") || check1(L->attn_q_norm, d->head_dim, "attn_q_norm") || check1(L->attn_k_norm, d->head_dim, "attn_k_norm")) goto fail;
            n_attn++;
        }
    }
    d->n_attn_layer = n_attn;
    d->n_recr_layer = n_recr;
    m->n_ctx = n_ctx;
    const size_t kv_elems = (size_t)n_ctx * d->n_kv_head * d->head_dim;
    const size_t conv_elems = (size_t)d->conv_dim * d->d_conv;
    const size_t ssm_elems = (size_t)d->n_v_heads * d->head_v_dim * d->head_k_dim;
    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        if (L->is_recurrent) {
            L->conv_state = alloc64(conv_elems * sizeof(float));
            L->ssm_state = alloc64(ssm_elems * sizeof(float));
            if (!L->conv_state || !L->ssm_state) goto fail;
        } else {
            L->kv_k = alloc64(kv_elems * sizeof(float));
            L->kv_v = alloc64(kv_elems * sizeof(float));
            if (!L->kv_k || !L->kv_v) goto fail;
        }
    }
    size_t stride_qkv = d->conv_dim;
    if (2 * d->n_head * d->head_dim > stride_qkv) stride_qkv = 2 * d->n_head * d->head_dim;
    if (d->n_ff > stride_qkv) stride_qkv = d->n_ff;
    const size_t floats = d->n_embd + d->n_embd + stride_qkv + 2 * d->n_kv_head * d->head_dim + d->d_inner + d->n_ff + d->n_head * d->head_dim + (size_t)c->vocab_size + (size_t)n_ctx;
    m->arena = alloc64(floats * sizeof(float));
    if (!m->arena) goto fail;
    {
        float *p = m->arena;
        m->x = p; p += d->n_embd;
        m->xb = p; p += d->n_embd;
        m->qkv = p; p += stride_qkv;
        m->kv_small = p; p += 2 * d->n_kv_head * d->head_dim;
        m->z = p; p += d->d_inner;
        m->ffn_buf = p; p += d->n_ff;
        m->attn_out = p; p += d->n_head * d->head_dim;
        m->logits = p; p += c->vocab_size;
        m->scores = p;
    }
    m->pos = 0;
    return 0;
fail:
    model_free(m);
    return -1;
}

void model_free(qwen_model_t *m) {
    if (!m) return;
    if (m->layers) {
        for (uint32_t il = 0; il < m->dims.n_layer; il++) {
            qwen_layer_t *L = &m->layers[il];
            free(L->kv_k);
            free(L->kv_v);
            free(L->conv_state);
            free(L->ssm_state);
        }
        free(m->layers);
    }
    free(m->arena);
    gguf_close(&m->gguf);
    memset(m, 0, sizeof *m);
}

static void apply_rope(float *vec, uint32_t pos, float theta,
                       const uint32_t *sections, uint32_t rope_dims) {
    uint32_t pair_offset = 0;
    for (int s = 0; s < 3; s++) {
        uint32_t sec_len = sections[s];
        for (uint32_t j = 0; j < sec_len; j++) {
            float freq = powf(theta, -2.0f * (float)j / (float)rope_dims);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            uint32_t idx0 = (pair_offset + j) * 2;
            uint32_t idx1 = idx0 + 1;
            float v0 = vec[idx0];
            float v1 = vec[idx1];
            vec[idx0] = v0 * cos_a - v1 * sin_a;
            vec[idx1] = v0 * sin_a + v1 * cos_a;
        }
        pair_offset += sec_len;
    }
}

void attention_layer(qwen_model_t *m, qwen_layer_t *L, pool_t *pool, uint32_t pos) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;
    const uint32_t head_dim = d->head_dim;
    const uint32_t n_head = d->n_head;
    const uint32_t n_kv_head = d->n_kv_head;
    const uint32_t n_embd = d->n_embd;
    const uint32_t n_ff = d->n_ff;

    rmsnorm(m->xb, m->x, (const float *)L->attn_norm->data, n_embd, eps);
    gemv(pool, L->attn_q, m->xb, m->qkv);
    gemv(pool, L->attn_k, m->xb, m->kv_small);
    gemv(pool, L->attn_v, m->xb, m->kv_small + n_kv_head * head_dim);

    const float *q_norm_w = (const float *)L->attn_q_norm->data;
    const float *k_norm_w = (const float *)L->attn_k_norm->data;
    for (uint32_t h = 0; h < n_head; h++) {
        float *qh = m->qkv + h * (2 * head_dim);
        rmsnorm(qh, qh, q_norm_w, head_dim, eps);
    }
    for (uint32_t kh = 0; kh < n_kv_head; kh++) {
        float *kh_ptr = m->kv_small + kh * head_dim;
        rmsnorm(kh_ptr, kh_ptr, k_norm_w, head_dim, eps);
    }

    float *k_cache_pos = L->kv_k + (size_t)pos * (n_kv_head * head_dim);
    float *v_cache_pos = L->kv_v + (size_t)pos * (n_kv_head * head_dim);
    memcpy(k_cache_pos, m->kv_small, (size_t)n_kv_head * head_dim * sizeof(float));
    memcpy(v_cache_pos, m->kv_small + n_kv_head * head_dim, (size_t)n_kv_head * head_dim * sizeof(float));

    for (uint32_t h = 0; h < n_head; h++) {
        float *qh = m->qkv + h * (2 * head_dim);
        apply_rope(qh, pos, m->cfg.rope_theta, d->rope_sections, m->cfg.rope_dim_count);
    }
    for (uint32_t kh = 0; kh < n_kv_head; kh++) {
        float *kh_ptr = k_cache_pos + kh * head_dim;
        apply_rope(kh_ptr, pos, m->cfg.rope_theta, d->rope_sections, m->cfg.rope_dim_count);
    }

    const float scale_attn = 1.0f / sqrtf((float)head_dim);
    const uint32_t heads_per_kv = n_head / n_kv_head;
    for (uint32_t h = 0; h < n_head; h++) {
        const uint32_t kh = h / heads_per_kv;
        const float *qh = m->qkv + h * (2 * head_dim);
        float *gate_h = m->qkv + h * (2 * head_dim) + head_dim;
        float *out_h = m->attn_out + h * head_dim;

        for (uint32_t t = 0; t <= pos; t++) {
            const float *kt = L->kv_k + (size_t)t * (n_kv_head * head_dim) + kh * head_dim;
            __m256 acc = _mm256_setzero_ps();
            for (uint32_t i = 0; i < head_dim; i += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_loadu_ps(kt + i), acc);
            }
            m->scores[t] = hsum256(acc) * scale_attn;
        }

        softmax_inplace(m->scores, pos + 1);

        memset(out_h, 0, head_dim * sizeof(float));
        for (uint32_t t = 0; t <= pos; t++) {
            const float st = m->scores[t];
            const float *vt = L->kv_v + (size_t)t * (n_kv_head * head_dim) + kh * head_dim;
            __m256 vst = _mm256_set1_ps(st);
            for (uint32_t i = 0; i < head_dim; i += 8) {
                __m256 vout = _mm256_loadu_ps(out_h + i);
                __m256 vval = _mm256_loadu_ps(vt + i);
                _mm256_storeu_ps(out_h + i, _mm256_fmadd_ps(vst, vval, vout));
            }
        }

        sigmoid_inplace(gate_h, head_dim);
        for (uint32_t i = 0; i < head_dim; i += 8) {
            __m256 vo = _mm256_loadu_ps(out_h + i);
            __m256 vg = _mm256_loadu_ps(gate_h + i);
            _mm256_storeu_ps(out_h + i, _mm256_mul_ps(vo, vg));
        }
    }

    gemv(pool, L->attn_out, m->attn_out, m->xb);
    for (uint32_t i = 0; i < n_embd; i += 8) {
        _mm256_storeu_ps(m->x + i, _mm256_add_ps(_mm256_loadu_ps(m->x + i), _mm256_loadu_ps(m->xb + i)));
    }

    rmsnorm(m->xb, m->x, (const float *)L->post_attn_norm->data, n_embd, eps);

    gemv(pool, L->ffn_gate, m->xb, m->ffn_buf);
    silu_inplace(m->ffn_buf, n_ff);
    gemv(pool, L->ffn_up, m->xb, m->qkv);
    for (uint32_t i = 0; i < n_ff; i += 8) {
        _mm256_storeu_ps(m->ffn_buf + i, _mm256_mul_ps(_mm256_loadu_ps(m->ffn_buf + i), _mm256_loadu_ps(m->qkv + i)));
    }
    gemv(pool, L->ffn_down, m->ffn_buf, m->xb);
    for (uint32_t i = 0; i < n_embd; i += 8) {
        _mm256_storeu_ps(m->x + i, _mm256_add_ps(_mm256_loadu_ps(m->x + i), _mm256_loadu_ps(m->xb + i)));
    }
}

void gdn_layer(qwen_model_t *m, qwen_layer_t *L, pool_t *pool) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;
    const uint32_t n_embd = d->n_embd;
    const uint32_t conv_dim = d->conv_dim;
    const uint32_t d_inner = d->d_inner;
    const uint32_t n_v_heads = d->n_v_heads;
    const uint32_t n_k_heads = d->n_k_heads;
    const uint32_t head_k_dim = d->head_k_dim;
    const uint32_t head_v_dim = d->head_v_dim;
    const uint32_t n_ff = d->n_ff;

    rmsnorm(m->xb, m->x, (const float *)L->attn_norm->data, n_embd, eps);
    gemv(pool, L->wqkv, m->xb, m->qkv);

    const float *conv_w = (const float *)L->ssm_conv1d->data;
    for (uint32_t c = 0; c < conv_dim; c++) {
        float *c_state = L->conv_state + c * 4;
        c_state[0] = c_state[1];
        c_state[1] = c_state[2];
        c_state[2] = c_state[3];
        c_state[3] = m->qkv[c];
        const float *cw = conv_w + c * 4;
        m->qkv[c] = c_state[0] * cw[0] + c_state[1] * cw[1] + c_state[2] * cw[2] + c_state[3] * cw[3];
    }
    silu_inplace(m->qkv, conv_dim);

    float *Q = m->qkv;
    float *K = m->qkv + n_k_heads * head_k_dim;
    float *V = m->qkv + 2 * n_k_heads * head_k_dim;

    for (uint32_t kh = 0; kh < n_k_heads; kh++) {
        l2norm(Q + kh * head_k_dim, Q + kh * head_k_dim, head_k_dim, eps);
        l2norm(K + kh * head_k_dim, K + kh * head_k_dim, head_k_dim, eps);
    }

    gemv(pool, L->wqkv_gate, m->xb, m->z);

    float beta[256], alpha[256];   /* bornés par n_v_heads (48 ici) */
    if (n_v_heads > 256) return;
    gemv(pool, L->ssm_beta, m->xb, beta);
    gemv(pool, L->ssm_alpha, m->xb, alpha);

    const float *dt_bias = (const float *)L->ssm_dt_bias->data;
    const float *ssm_a = (const float *)L->ssm_a->data;
    const float q_scale = 1.0f / sqrtf((float)head_k_dim);

    for (uint32_t h = 0; h < n_v_heads; h++) {
        const uint32_t kh = h % n_k_heads;
        const float *qh = Q + kh * head_k_dim;
        const float *k_vec = K + kh * head_k_dim;
        const float *v_vec = V + h * head_v_dim;
        float *S = L->ssm_state + (size_t)h * (head_v_dim * head_k_dim);
        float *out_h = m->attn_out + h * head_v_dim;

        float b_val = 1.0f / (1.0f + expf(-beta[h]));
        float a_sp = (alpha[h] + dt_bias[h] > 30.0f) ? (alpha[h] + dt_bias[h]) : log1pf(expf(alpha[h] + dt_bias[h]));
        float decay = expf(ssm_a[h] * a_sp);

        __m256 vdecay = _mm256_set1_ps(decay);
        for (uint32_t i = 0; i < head_v_dim * head_k_dim; i += 8) {
            _mm256_storeu_ps(S + i, _mm256_mul_ps(_mm256_loadu_ps(S + i), vdecay));
        }

        float delta[256], d_vec[256];   /* bornés par head_v_dim (128 ici) */
        for (uint32_t j = 0; j < head_k_dim; j++) {
            __m256 acc = _mm256_setzero_ps();
            for (uint32_t i = 0; i < head_v_dim; i += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(S + j * head_v_dim + i), _mm256_loadu_ps(k_vec + i), acc);
            }
            delta[j] = hsum256(acc);
        }

        for (uint32_t j = 0; j < head_v_dim; j++) {
            d_vec[j] = (v_vec[j] - delta[j]) * b_val;
        }

        for (uint32_t j = 0; j < head_k_dim; j++) {
            __m256 vd = _mm256_set1_ps(d_vec[j]);
            float *S_row = S + j * head_v_dim;
            for (uint32_t i = 0; i < head_v_dim; i += 8) {
                _mm256_storeu_ps(S_row + i, _mm256_fmadd_ps(_mm256_loadu_ps(k_vec + i), vd, _mm256_loadu_ps(S_row + i)));
            }
        }

        for (uint32_t j = 0; j < head_v_dim; j++) {
            __m256 acc = _mm256_setzero_ps();
            for (uint32_t i = 0; i < head_k_dim; i += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(S + j * head_v_dim + i), _mm256_loadu_ps(qh + i), acc);
            }
            out_h[j] = hsum256(acc) * q_scale;
        }
    }

    const float *ssm_norm_w = (const float *)L->ssm_norm->data;
    silu_inplace(m->z, d_inner);

    for (uint32_t h = 0; h < n_v_heads; h++) {
        float *out_h = m->attn_out + h * head_v_dim;
        const float *zh = m->z + h * head_v_dim;
        rmsnorm(out_h, out_h, ssm_norm_w, head_v_dim, eps);
        for (uint32_t i = 0; i < head_v_dim; i += 8) {
            _mm256_storeu_ps(out_h + i, _mm256_mul_ps(_mm256_loadu_ps(out_h + i), _mm256_loadu_ps(zh + i)));
        }
    }

    gemv(pool, L->ssm_out, m->attn_out, m->xb);
    for (uint32_t i = 0; i < n_embd; i += 8) {
        _mm256_storeu_ps(m->x + i, _mm256_add_ps(_mm256_loadu_ps(m->x + i), _mm256_loadu_ps(m->xb + i)));
    }

    rmsnorm(m->xb, m->x, (const float *)L->post_attn_norm->data, n_embd, eps);

    gemv(pool, L->ffn_gate, m->xb, m->ffn_buf);
    silu_inplace(m->ffn_buf, n_ff);
    gemv(pool, L->ffn_up, m->xb, m->qkv);
    for (uint32_t i = 0; i < n_ff; i += 8) {
        _mm256_storeu_ps(m->ffn_buf + i, _mm256_mul_ps(_mm256_loadu_ps(m->ffn_buf + i), _mm256_loadu_ps(m->qkv + i)));
    }
    gemv(pool, L->ffn_down, m->ffn_buf, m->xb);
    for (uint32_t i = 0; i < n_embd; i += 8) {
        _mm256_storeu_ps(m->x + i, _mm256_add_ps(_mm256_loadu_ps(m->x + i), _mm256_loadu_ps(m->xb + i)));
    }
}

void model_forward(qwen_model_t *m, pool_t *pool, uint32_t token, uint32_t pos) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;

    dequant_row(m->tok_embd->type, quant_row_ptr(m->tok_embd, token), d->n_embd, m->x);

    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        if (L->is_recurrent) {
            gdn_layer(m, L, pool);
        } else {
            attention_layer(m, L, pool, pos);
        }
    }

    rmsnorm(m->xb, m->x, (const float *)m->output_norm->data, d->n_embd, eps);
    gemv(pool, m->output, m->xb, m->logits);

    m->pos = pos + 1;
}

uint32_t model_sample_greedy(const float *logits, uint32_t vocab_size) {
    uint32_t best_id = 0;
    float max_logit = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best_id = i;
        }
    }
    return best_id;
}
