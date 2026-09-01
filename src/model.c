#define _POSIX_C_SOURCE 200809L

#include "model.h"
#include "kernels.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
    /* bornes des buffers de tete du forward (beta/alpha, delta/d_vec) */
    if (d->n_v_heads > 256 || d->head_v_dim > 256 || d->head_k_dim > 256) goto fail;

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

    /* Precomputed RoPE tables: cos/sin for every position and pair */
    {
        const uint32_t half = c->rope_dim_count / 2;
        m->rope_cos = alloc64((size_t)n_ctx * half * sizeof(float));
        m->rope_sin = alloc64((size_t)n_ctx * half * sizeof(float));
        if (!m->rope_cos || !m->rope_sin) goto fail;
        for (uint32_t p = 0; p < n_ctx; p++) {
            float *cs = m->rope_cos + (size_t)p * half;
            float *sn = m->rope_sin + (size_t)p * half;
            for (uint32_t i = 0; i < half; i++) {
                float angle = (float)p * powf(c->rope_theta, -2.0f * (float)i / (float)c->rope_dim_count);
                cs[i] = cosf(angle);
                sn[i] = sinf(angle);
            }
        }
    }

    /* ---- Batch buffers: speculative verification (B <= SPEC_MAX_B)
     *      and batched prefill chunks (B <= PREFILL_MAX_B) ---- */
    {
        const size_t B = PREFILL_MAX_B;   /* >= SPEC_MAX_B */
        const size_t stride_qkv = d->n_ff;   /* max(12288, 10240, 17408) = n_ff */
        const size_t floats =
            B * d->n_embd        /* bx */
          + B * d->n_embd        /* bxb */
          + B * stride_qkv       /* bqkv */
          + B * 2 * d->n_kv_head * d->head_dim  /* bkv */
          + B * d->d_inner       /* bz */
          + B * d->n_ff          /* bffn */
          + B * d->n_head * d->head_dim  /* battn_out (aussi n_v_heads*head_v = 6144) */
          + B * d->n_embd;       /* h_save */
        m->arena_b = alloc64(floats * sizeof(float));
        m->logits_b = alloc64(B * (size_t)c->vocab_size * sizeof(float));
        m->mtp_h = alloc64(d->n_embd * sizeof(float));
        if (!m->arena_b || !m->logits_b || !m->mtp_h) goto fail;
        float *p = m->arena_b;
        m->bx = p;        p += B * d->n_embd;
        m->bxb = p;       p += B * d->n_embd;
        m->bqkv = p;      p += B * stride_qkv;
        m->bkv = p;       p += B * 2 * d->n_kv_head * d->head_dim;
        m->bz = p;        p += B * d->d_inner;
        m->bffn = p;      p += B * d->n_ff;
        m->battn_out = p; p += B * d->n_head * d->head_dim;
        m->h_save = p;
    }

    /* ---- Recurrent state checkpoints for rollbacks ---- */
    m->ckpt_conv = alloc64((SPEC_MAX_B - 1) * (size_t)d->n_recr_layer
                           * d->conv_dim * d->d_conv * sizeof(float));
    m->ckpt_ssm  = alloc64((SPEC_MAX_B - 1) * (size_t)d->n_recr_layer
                           * d->n_v_heads * d->head_v_dim * d->head_k_dim * sizeof(float));
    if (!m->ckpt_conv || !m->ckpt_ssm) goto fail;

    /* ---- MTP Layer (blk.{n_layer}.* = nextn) ---- */
    {
        qwen_layer_t *L = &m->mtp_layer;
        memset(L, 0, sizeof *L);
        L->is_recurrent = false;
        const uint32_t il = d->n_layer;   /* la couche MTP = blk.{n_layer_tronc} */
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", il);
        if (!(L->attn_norm = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.post_attention_norm.weight", il);
        if (!(L->post_attn_norm = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.attn_q.weight", il);
        if (!(L->attn_q = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.attn_k.weight", il);
        if (!(L->attn_k = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.attn_v.weight", il);
        if (!(L->attn_v = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.attn_q_norm.weight", il);
        if (!(L->attn_q_norm = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.attn_k_norm.weight", il);
        if (!(L->attn_k_norm = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.attn_output.weight", il);
        if (!(L->attn_out = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.ffn_gate.weight", il);
        if (!(L->ffn_gate = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", il);
        if (!(L->ffn_up = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", il);
        if (!(L->ffn_down = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.nextn.eh_proj.weight", il);
        if (!(m->mtp_eh_proj = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.nextn.enorm.weight", il);
        if (!(m->mtp_enorm = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.nextn.hnorm.weight", il);
        if (!(m->mtp_hnorm = gguf_find_tensor(&m->gguf, name))) goto fail;
        snprintf(name, sizeof name, "blk.%u.nextn.shared_head_norm.weight", il);
        if (!(m->mtp_shnorm = gguf_find_tensor(&m->gguf, name))) goto fail;

        if (check2(m->mtp_eh_proj, 2 * d->n_embd, d->n_embd, "nextn.eh_proj") ||
            check1(m->mtp_enorm, d->n_embd, "nextn.enorm") ||
            check1(m->mtp_hnorm, d->n_embd, "nextn.hnorm") ||
            check1(m->mtp_shnorm, d->n_embd, "nextn.shared_head_norm")) goto fail;

        L->kv_k = alloc64(kv_elems * sizeof(float));
        L->kv_v = alloc64(kv_elems * sizeof(float));
        if (!L->kv_k || !L->kv_v) goto fail;
    }

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
    free(m->rope_cos);
    free(m->rope_sin);
    free(m->arena_b);
    free(m->logits_b);
    free(m->mtp_h);
    free(m->ckpt_conv);
    free(m->ckpt_ssm);
    free(m->mtp_layer.kv_k);
    free(m->mtp_layer.kv_v);
    gguf_close(&m->gguf);
    memset(m, 0, sizeof *m);
}

/* RoPE "IMROPE" (qwen35) en mode texte : NeoX PARTIEL, verifie sur ggml :
 *   - paires en demi-blocs : (i, i + rope_dims/2) sur les rope_dims premieres
 *     dims de la tete (rotate_pairs(n_dims, n_dims/2), PAS adjacentes)
 *   - index de frequence CONTINU 0..rope_dims/2-1 (pas de redemarrage)
 * cos/sin précalculés par position dans m->rope_cos/rope_sin. */
static void apply_rope(float *vec, uint32_t pos, const qwen_model_t *m) {
    const uint32_t half = m->cfg.rope_dim_count / 2;
    const float *cs = m->rope_cos + (size_t)pos * half;
    const float *sn = m->rope_sin + (size_t)pos * half;
    for (uint32_t i = 0; i < half; i++) {
        float v0 = vec[i];
        float v1 = vec[i + half];
        vec[i] = v0 * cs[i] - v1 * sn[i];
        vec[i + half] = v0 * sn[i] + v1 * cs[i];
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
        apply_rope(qh, pos, m);
    }
    for (uint32_t kh = 0; kh < n_kv_head; kh++) {
        float *kh_ptr = k_cache_pos + kh * head_dim;
        apply_rope(kh_ptr, pos, m);
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

    float beta[256], alpha[256];   /* bornes validees a l'init */
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

        /* k.q : constant par tete (pour la sortie algebrique) */
        __m256 kq_acc = _mm256_setzero_ps();
        for (uint32_t i = 0; i < head_k_dim; i += 8) {
            kq_acc = _mm256_fmadd_ps(_mm256_loadu_ps(k_vec + i), _mm256_loadu_ps(qh + i), kq_acc);
        }
        const float kq = hsum256(kq_acc);

        /* Fusion exacte des 4 balayages de S en 2 passes par ligne :
         *   delta_j = decay * (S_j . k)
         *   d_j     = (v_j - delta_j) * beta
         *   S'_j    = decay * S_j + k * d_j
         *   o_j     = decay * (S_j . q) + d_j * (k . q)
         * Passe 1 : les deux dots par ligne ; passe 2 : ecriture de S'. */
        for (uint32_t j = 0; j < head_k_dim; j++) {
            const float *row = S + (size_t)j * head_v_dim;
            __m256 acc_k = _mm256_setzero_ps();
            __m256 acc_q = _mm256_setzero_ps();
            for (uint32_t i = 0; i < head_v_dim; i += 8) {
                __m256 srow = _mm256_loadu_ps(row + i);
                acc_k = _mm256_fmadd_ps(srow, _mm256_loadu_ps(k_vec + i), acc_k);
                acc_q = _mm256_fmadd_ps(srow, _mm256_loadu_ps(qh + i), acc_q);
            }
            float delta = decay * hsum256(acc_k);
            float d_j = (v_vec[j] - delta) * b_val;
            out_h[j] = (decay * hsum256(acc_q) + d_j * kq) * q_scale;

            float *row_w = S + (size_t)j * head_v_dim;
            __m256 vdj = _mm256_set1_ps(d_j);
            __m256 vdec = _mm256_set1_ps(decay);
            for (uint32_t i = 0; i < head_v_dim; i += 8) {
                __m256 srow = _mm256_loadu_ps(row_w + i);
                srow = _mm256_fmadd_ps(vdj, _mm256_loadu_ps(k_vec + i), _mm256_mul_ps(vdec, srow));
                _mm256_storeu_ps(row_w + i, srow);
            }
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

/* profil léger : cumuls sur le thread appelant uniquement */
double qwen_prof_embed, qwen_prof_gdn, qwen_prof_attn, qwen_prof_head;
uint64_t qwen_prof_calls;

#ifdef QWEN_PROFILE
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#define PROF_T0 _t0 = now_sec();
#define PROF_ACC(x) x += now_sec() - _t0;
#else
#define PROF_T0
#define PROF_ACC(x)
#endif

void model_forward(qwen_model_t *m, pool_t *pool, uint32_t token, uint32_t pos) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;
    double _t0 = 0.0;
    (void)_t0;

    PROF_T0
    dequant_row(m->tok_embd->type, quant_row_ptr(m->tok_embd, token), d->n_embd, m->x);
    PROF_ACC(qwen_prof_embed)

    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        if (L->is_recurrent) {
            PROF_T0
            gdn_layer(m, L, pool);
            PROF_ACC(qwen_prof_gdn)
        } else {
            PROF_T0
            attention_layer(m, L, pool, pos);
            PROF_ACC(qwen_prof_attn)
        }
    }

    PROF_T0
    rmsnorm(m->xb, m->x, (const float *)m->output_norm->data, d->n_embd, eps);
    gemv(pool, m->output, m->xb, m->logits);
    PROF_ACC(qwen_prof_head)

    qwen_prof_calls++;
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

/* ============== Decodage speculatif : couches batchees ============== */

static void attention_layer_batch(qwen_model_t *m, qwen_layer_t *L, pool_t *pool,
                                  uint32_t pos, uint32_t B) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;
    const uint32_t head_dim = d->head_dim;
    const uint32_t n_head = d->n_head;
    const uint32_t n_kv_head = d->n_kv_head;
    const uint32_t n_embd = d->n_embd;
    const uint32_t n_ff = d->n_ff;
    const uint32_t kvrow = n_kv_head * head_dim;           /* 1024 */
    const uint32_t stride = n_ff;                          /* stride bqkv */

    for (uint32_t b = 0; b < B; b++)
        rmsnorm(m->bxb + b * n_embd, m->bx + b * n_embd,
                (const float *)L->attn_norm->data, n_embd, eps);
    gemv_batch(pool, L->attn_q, m->bxb, n_embd, B, m->bqkv, stride);
    gemv_batch(pool, L->attn_k, m->bxb, n_embd, B, m->bkv, 2 * kvrow);
    gemv_batch(pool, L->attn_v, m->bxb, n_embd, B, m->bkv + kvrow, 2 * kvrow);

    const float *q_norm_w = (const float *)L->attn_q_norm->data;
    const float *k_norm_w = (const float *)L->attn_k_norm->data;
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *qh = m->bqkv + b * stride + h * (2 * head_dim);
            rmsnorm(qh, qh, q_norm_w, head_dim, eps);
        }
        for (uint32_t kh = 0; kh < n_kv_head; kh++) {
            float *khp = m->bkv + b * 2 * kvrow + kh * head_dim;
            rmsnorm(khp, khp, k_norm_w, head_dim, eps);
        }
        float *k_cache_pos = L->kv_k + (size_t)(pos + b) * kvrow;
        float *v_cache_pos = L->kv_v + (size_t)(pos + b) * kvrow;
        memcpy(k_cache_pos, m->bkv + b * 2 * kvrow, kvrow * sizeof(float));
        memcpy(v_cache_pos, m->bkv + b * 2 * kvrow + kvrow, kvrow * sizeof(float));
    }

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < n_head; h++)
            apply_rope(m->bqkv + b * stride + h * (2 * head_dim), pos + b, m);
        for (uint32_t kh = 0; kh < n_kv_head; kh++)
            apply_rope(L->kv_k + (size_t)(pos + b) * kvrow + kh * head_dim, pos + b, m);
    }

    const float scale_attn = 1.0f / sqrtf((float)head_dim);
    const uint32_t heads_per_kv = n_head / n_kv_head;
    for (uint32_t h = 0; h < n_head; h++) {
        const uint32_t kh = h / heads_per_kv;

        for (uint32_t b = 0; b < B; b++) {
            const float *qh = m->bqkv + b * stride + h * (2 * head_dim);
            float *gate_h = m->bqkv + b * stride + h * (2 * head_dim) + head_dim;
            float *out_h = m->battn_out + b * (n_head * head_dim) + h * head_dim;
            const uint32_t pl = pos + b;

            for (uint32_t t = 0; t <= pl; t++) {
                const float *kt = L->kv_k + (size_t)t * kvrow + kh * head_dim;
                __m256 acc = _mm256_setzero_ps();
                for (uint32_t i = 0; i < head_dim; i += 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_loadu_ps(kt + i), acc);
                }
                m->scores[t] = hsum256(acc) * scale_attn;
            }

            softmax_inplace(m->scores, pl + 1);

            memset(out_h, 0, head_dim * sizeof(float));
            for (uint32_t t = 0; t <= pl; t++) {
                const float st = m->scores[t];
                const float *vt = L->kv_v + (size_t)t * kvrow + kh * head_dim;
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
    }

    gemv_batch(pool, L->attn_out, m->battn_out, n_head * head_dim, B, m->bxb, n_embd);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t i = 0; i < n_embd; i += 8)
            _mm256_storeu_ps(m->bx + b * n_embd + i,
                _mm256_add_ps(_mm256_loadu_ps(m->bx + b * n_embd + i),
                              _mm256_loadu_ps(m->bxb + b * n_embd + i)));

    for (uint32_t b = 0; b < B; b++)
        rmsnorm(m->bxb + b * n_embd, m->bx + b * n_embd,
                (const float *)L->post_attn_norm->data, n_embd, eps);

    gemv_batch(pool, L->ffn_gate, m->bxb, n_embd, B, m->bffn, n_ff);
    for (uint32_t b = 0; b < B; b++)
        silu_inplace(m->bffn + b * n_ff, n_ff);
    gemv_batch(pool, L->ffn_up, m->bxb, n_embd, B, m->bqkv, stride);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t i = 0; i < n_ff; i += 8)
            _mm256_storeu_ps(m->bffn + b * n_ff + i,
                _mm256_mul_ps(_mm256_loadu_ps(m->bffn + b * n_ff + i),
                              _mm256_loadu_ps(m->bqkv + b * stride + i)));
    gemv_batch(pool, L->ffn_down, m->bffn, n_ff, B, m->bxb, n_embd);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t i = 0; i < n_embd; i += 8)
            _mm256_storeu_ps(m->bx + b * n_embd + i,
                _mm256_add_ps(_mm256_loadu_ps(m->bx + b * n_embd + i),
                              _mm256_loadu_ps(m->bxb + b * n_embd + i)));
}

static void gdn_layer_batch(qwen_model_t *m, qwen_layer_t *L, pool_t *pool,
                            uint32_t B, int ckpt) {
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
    const uint32_t stride = n_ff;   /* stride bqkv >= conv_dim */

    for (uint32_t b = 0; b < B; b++)
        rmsnorm(m->bxb + b * n_embd, m->bx + b * n_embd,
                (const float *)L->attn_norm->data, n_embd, eps);
    gemv_batch(pool, L->wqkv, m->bxb, n_embd, B, m->bqkv, stride);

    /* conv1d causale : sequentielle sur le batch (l'etat glisse).
     * Checkpoints conv/SSM uniquement en mode verification speculative :
     * index de cette couche parmi les recurrentes + offsets dans ckpt_*.
     * n_ckpt borne le nombre de checkpoints a la capacite des buffers
     * (SPEC_MAX_B - 1 slots) meme si B > SPEC_MAX_B. */
    const float *conv_w = (const float *)L->ssm_conv1d->data;
    const size_t conv_elems = (size_t)conv_dim * d->d_conv;
    const size_t ssm_elems  = (size_t)n_v_heads * head_v_dim * head_k_dim;
    const uint32_t n_ckpt = (B < SPEC_MAX_B) ? B : SPEC_MAX_B;
    size_t layer_off_conv = 0, layer_off_ssm = 0;
    if (ckpt) {
        uint32_t ir = 0;
        for (uint32_t i = 0; i < d->n_layer; i++) {
            if (m->layers[i].is_recurrent) {
                if (&m->layers[i] == L) break;
                ir++;
            }
        }
        layer_off_conv = (size_t)ir * conv_elems;
        layer_off_ssm  = (size_t)ir * ssm_elems;
    }

    {
        for (uint32_t b = 0; b < B; b++) {
            float *xb_qkv = m->bqkv + b * stride;
            for (uint32_t c = 0; c < conv_dim; c++) {
                float *c_state = L->conv_state + c * 4;
                c_state[0] = c_state[1];
                c_state[1] = c_state[2];
                c_state[2] = c_state[3];
                c_state[3] = xb_qkv[c];
                const float *cw = conv_w + c * 4;
                xb_qkv[c] = c_state[0] * cw[0] + c_state[1] * cw[1]
                          + c_state[2] * cw[2] + c_state[3] * cw[3];
            }
            if (ckpt && b + 1 < n_ckpt) {
                memcpy(m->ckpt_conv + (size_t)b * m->dims.n_recr_layer * conv_elems
                         + layer_off_conv,
                       L->conv_state, conv_elems * sizeof(float));
            }
            silu_inplace(xb_qkv, conv_dim);

            float *Q = xb_qkv;
            float *K = xb_qkv + n_k_heads * head_k_dim;
            for (uint32_t kh = 0; kh < n_k_heads; kh++) {
                l2norm(Q + kh * head_k_dim, Q + kh * head_k_dim, head_k_dim, eps);
                l2norm(K + kh * head_k_dim, K + kh * head_k_dim, head_k_dim, eps);
            }
        }
    }

    gemv_batch(pool, L->wqkv_gate, m->bxb, n_embd, B, m->bz, d_inner);

    float beta_b[PREFILL_MAX_B][256], alpha_b[PREFILL_MAX_B][256];
    gemv_batch(pool, L->ssm_beta, m->bxb, n_embd, B, &beta_b[0][0], 256);
    gemv_batch(pool, L->ssm_alpha, m->bxb, n_embd, B, &alpha_b[0][0], 256);

    const float *dt_bias = (const float *)L->ssm_dt_bias->data;
    const float *ssm_a = (const float *)L->ssm_a->data;
    const float q_scale = 1.0f / sqrtf((float)head_k_dim);

    /* recurrence SSM : sequentielle sur le batch (l'etat "apres le token b"
     * pour la couche n'existe qu'une fois toutes les tetes a jour -> checkpoint
     * apres chaque token, en mode verification speculative uniquement) */
    {
        for (uint32_t b = 0; b < B; b++) {
            for (uint32_t h = 0; h < n_v_heads; h++) {
                const uint32_t kh = h % n_k_heads;
                float *S = L->ssm_state + (size_t)h * (head_v_dim * head_k_dim);
                const float *xb_qkv = m->bqkv + b * stride;
                const float *qh = xb_qkv + kh * head_k_dim;
                const float *k_vec = xb_qkv + n_k_heads * head_k_dim + kh * head_k_dim;
                const float *v_vec = xb_qkv + 2 * n_k_heads * head_k_dim + h * head_v_dim;
                float *out_h = m->battn_out + b * d_inner + h * head_v_dim;

                float b_val = 1.0f / (1.0f + expf(-beta_b[b][h]));
                float a_raw = alpha_b[b][h] + dt_bias[h];
                float a_sp = (a_raw > 30.0f) ? a_raw : log1pf(expf(a_raw));
                float decay = expf(ssm_a[h] * a_sp);

                __m256 kq_acc = _mm256_setzero_ps();
                for (uint32_t i = 0; i < head_k_dim; i += 8) {
                    kq_acc = _mm256_fmadd_ps(_mm256_loadu_ps(k_vec + i), _mm256_loadu_ps(qh + i), kq_acc);
                }
                const float kq = hsum256(kq_acc);

                for (uint32_t j = 0; j < head_k_dim; j++) {
                    const float *row = S + (size_t)j * head_v_dim;
                    __m256 acc_k = _mm256_setzero_ps();
                    __m256 acc_q = _mm256_setzero_ps();
                    for (uint32_t i = 0; i < head_v_dim; i += 8) {
                        __m256 srow = _mm256_loadu_ps(row + i);
                        acc_k = _mm256_fmadd_ps(srow, _mm256_loadu_ps(k_vec + i), acc_k);
                        acc_q = _mm256_fmadd_ps(srow, _mm256_loadu_ps(qh + i), acc_q);
                    }
                    float delta = decay * hsum256(acc_k);
                    float d_j = (v_vec[j] - delta) * b_val;
                    out_h[j] = (decay * hsum256(acc_q) + d_j * kq) * q_scale;

                    float *row_w = S + (size_t)j * head_v_dim;
                    __m256 vdj = _mm256_set1_ps(d_j);
                    __m256 vdec = _mm256_set1_ps(decay);
                    for (uint32_t i = 0; i < head_v_dim; i += 8) {
                        __m256 srow = _mm256_loadu_ps(row_w + i);
                        srow = _mm256_fmadd_ps(vdj, _mm256_loadu_ps(k_vec + i), _mm256_mul_ps(vdec, srow));
                        _mm256_storeu_ps(row_w + i, srow);
                    }
                }
            }

            /* checkpoint SSM apres le token b (sauf le dernier) ;
             * conv deja checkpointe dans sa propre boucle */
            if (ckpt && b + 1 < n_ckpt) {
                memcpy(m->ckpt_ssm + (size_t)b * m->dims.n_recr_layer * ssm_elems
                         + layer_off_ssm,
                       L->ssm_state, ssm_elems * sizeof(float));
            }
        }
    }

    const float *ssm_norm_w = (const float *)L->ssm_norm->data;
    for (uint32_t b = 0; b < B; b++)
        silu_inplace(m->bz + b * d_inner, d_inner);

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < n_v_heads; h++) {
            float *out_h = m->battn_out + b * d_inner + h * head_v_dim;
            const float *zh = m->bz + b * d_inner + h * head_v_dim;
            rmsnorm(out_h, out_h, ssm_norm_w, head_v_dim, eps);
            for (uint32_t i = 0; i < head_v_dim; i += 8) {
                _mm256_storeu_ps(out_h + i,
                    _mm256_mul_ps(_mm256_loadu_ps(out_h + i), _mm256_loadu_ps(zh + i)));
            }
        }
    }

    gemv_batch(pool, L->ssm_out, m->battn_out, d_inner, B, m->bxb, n_embd);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t i = 0; i < n_embd; i += 8)
            _mm256_storeu_ps(m->bx + b * n_embd + i,
                _mm256_add_ps(_mm256_loadu_ps(m->bx + b * n_embd + i),
                              _mm256_loadu_ps(m->bxb + b * n_embd + i)));

    for (uint32_t b = 0; b < B; b++)
        rmsnorm(m->bxb + b * n_embd, m->bx + b * n_embd,
                (const float *)L->post_attn_norm->data, n_embd, eps);

    gemv_batch(pool, L->ffn_gate, m->bxb, n_embd, B, m->bffn, n_ff);
    for (uint32_t b = 0; b < B; b++)
        silu_inplace(m->bffn + b * n_ff, n_ff);
    gemv_batch(pool, L->ffn_up, m->bxb, n_embd, B, m->bqkv, stride);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t i = 0; i < n_ff; i += 8)
            _mm256_storeu_ps(m->bffn + b * n_ff + i,
                _mm256_mul_ps(_mm256_loadu_ps(m->bffn + b * n_ff + i),
                              _mm256_loadu_ps(m->bqkv + b * stride + i)));
    gemv_batch(pool, L->ffn_down, m->bffn, n_ff, B, m->bxb, n_embd);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t i = 0; i < n_embd; i += 8)
            _mm256_storeu_ps(m->bx + b * n_embd + i,
                _mm256_add_ps(_mm256_loadu_ps(m->bx + b * n_embd + i),
                              _mm256_loadu_ps(m->bxb + b * n_embd + i)));
}

void model_forward_batch(qwen_model_t *m, pool_t *pool,
                         const uint32_t *tokens, uint32_t pos, uint32_t B, int ckpt) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;

    for (uint32_t b = 0; b < B; b++)
        dequant_row(m->tok_embd->type, quant_row_ptr(m->tok_embd, tokens[b]),
                    d->n_embd, m->bx + b * d->n_embd);

    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        if (L->is_recurrent)
            gdn_layer_batch(m, L, pool, B, ckpt);
        else
            attention_layer_batch(m, L, pool, pos, B);
    }

    /* etat cache final par token (entree MTP) : AVANT output_norm */
    memcpy(m->h_save, m->bx, (size_t)B * d->n_embd * sizeof(float));

    for (uint32_t b = 0; b < B; b++)
        rmsnorm(m->bxb + b * d->n_embd, m->bx + b * d->n_embd,
                (const float *)m->output_norm->data, d->n_embd, eps);
    gemv_batch(pool, m->output, m->bxb, d->n_embd, B, m->logits_b, m->cfg.vocab_size);

    m->pos = pos + B;
}

int model_snapshot_states(qwen_model_t *m) {
    (void)m;   /* remplace par les checkpoints par token du batch */
    return 0;
}

void model_rollback_states(qwen_model_t *m) {
    (void)m;
}

/* Restaure les etats recurrents tels qu'ils etaient APRES le token b
 * du dernier model_forward_batch (b < B-1). L'appellant ajuste m->pos. */
void model_rollback_to(qwen_model_t *m, uint32_t b) {
    const qwen_dims_t *d = &m->dims;
    const size_t conv_elems = (size_t)d->conv_dim * d->d_conv;
    const size_t ssm_elems = (size_t)d->n_v_heads * d->head_v_dim * d->head_k_dim;
    uint32_t ir = 0;
    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        if (!L->is_recurrent) continue;
        memcpy(L->conv_state,
               m->ckpt_conv + (size_t)b * d->n_recr_layer * conv_elems + ir * conv_elems,
               conv_elems * sizeof(float));
        memcpy(L->ssm_state,
               m->ckpt_ssm + (size_t)b * d->n_recr_layer * ssm_elems + ir * ssm_elems,
               ssm_elems * sizeof(float));
        ir++;
    }
}

/* Couche MTP (nextn) : MTP(t[q], h[q]) @ pos q -> h_nextn (+ logits).
 * Semantique verifiee sur llama.cpp src/models/qwen35.cpp graph_mtp :
 *   e = RMSNorm(embd(token), enorm) ; h' = RMSNorm(h, hnorm)
 *   x = eh_proj @ [e || h']  ; puis couche d'attention GATEE standard
 *   (+ residu, FFN) ; h_nextn = sortie ; logits = head @ RMSNorm(x, shnorm) */
int mtp_forward(qwen_model_t *m, pool_t *pool, uint32_t token, uint32_t pos,
                int want_logits) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;
    const uint32_t n_embd = d->n_embd;

    /* concat [e_norm || h_norm] dans m->qkv (10240 = 2 * n_embd) */
    dequant_row(m->tok_embd->type, quant_row_ptr(m->tok_embd, token), n_embd, m->qkv);
    rmsnorm(m->qkv, m->qkv, (const float *)m->mtp_enorm->data, n_embd, eps);
    rmsnorm(m->qkv + n_embd, m->mtp_h, (const float *)m->mtp_hnorm->data, n_embd, eps);

    /* m->x = eh_proj @ concat (residu inpSA) */
    gemv(pool, m->mtp_eh_proj, m->qkv, m->x);

    /* couche d'attention gatee standard (KV cache dedie, FFN inclus) */
    attention_layer(m, &m->mtp_layer, pool, pos);

    /* h_nextn pour le chainage / l'ancre du prochain cycle */
    memcpy(m->mtp_h, m->x, n_embd * sizeof(float));

    if (want_logits) {
        rmsnorm(m->xb, m->x, (const float *)m->mtp_shnorm->data, n_embd, eps);
        gemv(pool, m->output, m->xb, m->logits);
    }
    return 0;
}

/* MTP batche (prefill) : remplit le KV cache MTP pour tokens[pos..pos+B-1].
 * Entree h par token = h_save[b] (etat cache du backbone AVANT output_norm,
 * rempli par model_forward_batch) ; l'ancre mtp_h = sortie du DERNIER token.
 * Semantique identique a n appels a mtp_forward(..., want_logits=0). */
int mtp_forward_batch(qwen_model_t *m, pool_t *pool, const uint32_t *tokens,
                      uint32_t pos, uint32_t B, int want_logits) {
    const qwen_dims_t *d = &m->dims;
    const float eps = m->cfg.rms_norm_eps;
    const uint32_t n_embd = d->n_embd;
    const uint32_t stride = d->n_ff;   /* stride bqkv >= 2*n_embd */

    /* concat [e_norm || h_norm] par token dans bqkv[b*stride] */
    for (uint32_t b = 0; b < B; b++) {
        float *cat = m->bqkv + (size_t)b * stride;
        dequant_row(m->tok_embd->type, quant_row_ptr(m->tok_embd, tokens[b]),
                    n_embd, cat);
        rmsnorm(cat, cat, (const float *)m->mtp_enorm->data, n_embd, eps);
        rmsnorm(cat + n_embd, m->h_save + (size_t)b * n_embd,
                (const float *)m->mtp_hnorm->data, n_embd, eps);
    }

    /* bx = eh_proj @ concat (residu d'entree de la couche MTP) */
    gemv_batch(pool, m->mtp_eh_proj, m->bqkv, stride, B, m->bx, n_embd);

    /* couche d'attention gatee batchee (KV cache MTP dedie, FFN inclus) */
    attention_layer_batch(m, &m->mtp_layer, pool, pos, B);

    /* h_nextn : ancre = sortie du dernier token du chunk */
    memcpy(m->mtp_h, m->bx + (size_t)(B - 1) * n_embd, n_embd * sizeof(float));

    if (want_logits) {
        for (uint32_t b = 0; b < B; b++)
            rmsnorm(m->bxb + (size_t)b * n_embd, m->bx + (size_t)b * n_embd,
                    (const float *)m->mtp_shnorm->data, n_embd, eps);
        gemv_batch(pool, m->output, m->bxb, n_embd, B, m->logits_b,
                   m->cfg.vocab_size);
    }
    return 0;
}
