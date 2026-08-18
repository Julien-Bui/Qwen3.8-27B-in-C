#define _POSIX_C_SOURCE 200809L

#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Allocation alignée 64 octets (ligne de cache, accès AVX2 sans pénalité). */
static void *alloc64(size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, 64, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

static const gguf_tensor_info_t *must_find(const gguf_context_t *g, const char *name) {
    const gguf_tensor_info_t *t = gguf_find_tensor(g, name);
    if (!t) fprintf(stderr, "model_init: tenseur manquant '%s'\n", name);
    return t;
}

/* Vérifie qu'un tenseur 2D a exactement [ne0, ne1] (dans cet ordre GGUF :
 * ne[0] = dim de ligne = entrée de la projection). */
static int check2(const gguf_tensor_info_t *t, uint64_t ne0, uint64_t ne1, const char *what) {
    if (!t) return -1;
    if (t->n_dims != 2 || t->ne[0] != ne0 || t->ne[1] != ne1) {
        fprintf(stderr, "model_init: %s : attendu [%llu,%llu], trouvé [%llu,%llu]\n",
                what, (unsigned long long)ne0, (unsigned long long)ne1,
                (unsigned long long)t->ne[0], (unsigned long long)t->ne[1]);
        return -1;
    }
    return 0;
}

static int check1(const gguf_tensor_info_t *t, uint64_t ne0, const char *what) {
    if (!t) return -1;
    if (t->n_dims != 1 || t->ne[0] != ne0) {
        fprintf(stderr, "model_init: %s : attendu [%llu], trouvé [%llu]\n",
                what, (unsigned long long)ne0, (unsigned long long)t->ne[0]);
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

    /* ---- Dimensions dérivées, puis validées contre les tenseurs réels ---- */
    d->n_embd   = c->n_embd;
    d->n_ff     = c->n_ff;
    d->n_head   = c->n_head;
    d->n_kv_head = c->n_head_kv;
    d->head_dim = c->head_dim;
    d->n_k_heads = c->ssm_group_count;
    d->head_k_dim = c->ssm_state_size;
    d->head_v_dim = c->ssm_state_size;
    d->d_inner  = c->ssm_inner_size;
    d->d_conv   = c->ssm_conv_kernel;
    d->d_state  = c->ssm_state_size;
    memcpy(d->rope_sections, c->rope_sections, sizeof d->rope_sections);

    /* n_v_heads n'est pas dans nos métadonnées : lu sur ssm_a [n_v_heads] */
    char name[64];
    snprintf(name, sizeof name, "blk.0.ssm_a");
    const gguf_tensor_info_t *a0 = gguf_find_tensor(&m->gguf, name);
    if (!a0 || a0->n_dims != 1) {
        fprintf(stderr, "model_init: blk.0.ssm_a introuvable (dérivation n_v_heads)\n");
        goto fail;
    }
    d->n_v_heads = (uint32_t)a0->ne[0];
    d->conv_dim  = 2 * d->n_k_heads * d->head_k_dim + d->d_inner;

    /* Le tronc = n_layer - nextn (la/les couches MTP ne tournent pas en v1). */
    d->n_layer = (c->n_layer > c->nextn_layers) ? c->n_layer - c->nextn_layers : c->n_layer;

    /* Cohérence arithmétique globale avant toute allocation */
    if (d->d_inner != d->n_v_heads * d->head_v_dim) {
        fprintf(stderr, "model_init: d_inner (%u) != n_v_heads*head_v (%u*%u)\n",
                d->d_inner, d->n_v_heads, d->head_v_dim);
        goto fail;
    }

    m->tok_embd    = must_find(&m->gguf, "token_embd.weight");
    m->output_norm = must_find(&m->gguf, "output_norm.weight");
    m->output      = gguf_find_tensor(&m->gguf, "output.weight");
    if (!m->output) m->output = m->tok_embd;   /* tied embeddings */
    if (!m->tok_embd || !m->output_norm) goto fail;
    if (check2(m->tok_embd, d->n_embd, c->vocab_size, "token_embd")) goto fail;

    /* ---- Couches ---- */
    m->layers = calloc(d->n_layer, sizeof *m->layers);
    if (!m->layers) goto fail;

    uint32_t n_attn = 0, n_recr = 0;
    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        /* (i+1) % interval != 0 => récurrente (qwen35.cpp) */
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
        if (check2(L->ffn_gate, d->n_embd, d->n_ff, "ffn_gate") ||
            check2(L->ffn_up,   d->n_embd, d->n_ff, "ffn_up")   ||
            check2(L->ffn_down, d->n_ff,   d->n_embd, "ffn_down")) goto fail;

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

            if (check2(L->wqkv,      d->n_embd, d->conv_dim, "wqkv")      ||
                check2(L->wqkv_gate, d->n_embd, d->d_inner,  "wqkv_gate") ||
                check2(L->ssm_conv1d, d->d_conv, d->conv_dim, "ssm_conv1d") ||
                check2(L->ssm_beta,   d->n_embd, d->n_v_heads, "ssm_beta") ||
                check2(L->ssm_alpha,  d->n_embd, d->n_v_heads, "ssm_alpha") ||
                check2(L->ssm_out,    d->d_inner, d->n_embd,  "ssm_out")  ||
                check1(L->ssm_dt_bias, d->n_v_heads, "ssm_dt_bias")       ||
                check1(L->ssm_a,      d->n_v_heads, "ssm_a")              ||
                check1(L->ssm_norm,   d->head_v_dim, "ssm_norm")) goto fail;
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

            if (check2(L->attn_q, d->n_embd, 2 * d->n_head * d->head_dim, "attn_q") ||
                check2(L->attn_k, d->n_embd, d->n_kv_head * d->head_dim, "attn_k")  ||
                check2(L->attn_v, d->n_embd, d->n_kv_head * d->head_dim, "attn_v")  ||
                check2(L->attn_out, d->n_head * d->head_dim, d->n_embd, "attn_out") ||
                check1(L->attn_q_norm, d->head_dim, "attn_q_norm")                  ||
                check1(L->attn_k_norm, d->head_dim, "attn_k_norm")) goto fail;
            n_attn++;
        }
    }
    d->n_attn_layer = n_attn;
    d->n_recr_layer = n_recr;

    /* ---- Allocation des états ---- */
    m->n_ctx = n_ctx;
    const size_t kv_elems = (size_t)n_ctx * d->n_kv_head * d->head_dim;
    const size_t conv_elems = (size_t)d->conv_dim * d->d_conv;
    const size_t ssm_elems = (size_t)d->n_v_heads * d->head_v_dim * d->head_k_dim;

    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m->layers[il];
        if (L->is_recurrent) {
            L->conv_state = alloc64(conv_elems * sizeof(float));
            L->ssm_state  = alloc64(ssm_elems * sizeof(float));
            if (!L->conv_state || !L->ssm_state) goto fail;
        } else {
            L->kv_k = alloc64(kv_elems * sizeof(float));
            L->kv_v = alloc64(kv_elems * sizeof(float));
            if (!L->kv_k || !L->kv_v) goto fail;
        }
    }

    /* ---- Arène d'activations (une seule allocation alignée) ---- */
    const size_t stride_qkv = d->conv_dim > 2 * d->n_head * d->head_dim
                            ? d->conv_dim : 2 * d->n_head * d->head_dim;
    const size_t floats =
        d->n_embd        /* x */
      + d->n_embd        /* xb */
      + stride_qkv       /* qkv */
      + 2 * d->n_kv_head * d->head_dim /* kv_small */
      + d->d_inner       /* z */
      + d->n_ff          /* ffn_buf */
      + d->n_head * d->head_dim /* attn_out */
      + (size_t)c->vocab_size; /* logits */
    m->arena = alloc64(floats * sizeof(float));
    if (!m->arena) goto fail;
    {
        float *p = m->arena;
        m->x        = p; p += d->n_embd;
        m->xb       = p; p += d->n_embd;
        m->qkv      = p; p += stride_qkv;
        m->kv_small = p; p += 2 * d->n_kv_head * d->head_dim;
        m->z        = p; p += d->d_inner;
        m->ffn_buf  = p; p += d->n_ff;
        m->attn_out = p; p += d->n_head * d->head_dim;
        m->logits   = p;
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
