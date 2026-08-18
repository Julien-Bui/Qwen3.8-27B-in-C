#ifndef MODEL_H
#define MODEL_H

#include "gguf.h"
#include <stdint.h>
#include <stdbool.h>

/* Dimensions dérivées des tenseurs/métadonnées, validées par assertions
 * à l'init. Sémantique : src/models/qwen35.cpp de llama.cpp. */
typedef struct {
    /* commun */
    uint32_t n_embd;        /* 5120 */
    uint32_t n_ff;          /* 17408 */
    uint32_t n_layer;       /* 64 couches tronc (MTP exclue) */
    uint32_t n_attn_layer;  /* 16 couches pleine attention */
    uint32_t n_recr_layer;  /* 48 couches GDN */
    /* pleine attention (gated GQA) */
    uint32_t n_head;        /* 24 */
    uint32_t n_kv_head;     /* 4 */
    uint32_t head_dim;      /* 256 ; q projeté à 2x (q||gate entrelacés) */
    uint32_t rope_sections[4];
    /* Gated DeltaNet */
    uint32_t n_v_heads;     /* 48 (= dt_rank, lu sur ssm_a.ne[0]) */
    uint32_t n_k_heads;     /* 16 (= group_count) */
    uint32_t head_k_dim;    /* 128 */
    uint32_t head_v_dim;    /* 128 */
    uint32_t d_inner;       /* 6144 */
    uint32_t conv_dim;      /* 2*16*128 + 6144 = 10240 */
    uint32_t d_conv;        /* 4 */
    uint32_t d_state;       /* 128 */
} qwen_dims_t;

typedef struct {
    bool is_recurrent;      /* true = GDN, false = pleine attention */
    /* partagé */
    const gguf_tensor_info_t *attn_norm;       /* RMSNorm pré-attention */
    const gguf_tensor_info_t *post_attn_norm;  /* RMSNorm pré-FFN */
    const gguf_tensor_info_t *ffn_gate, *ffn_up, *ffn_down;
    /* pleine attention */
    const gguf_tensor_info_t *attn_q;          /* [n_embd, 2*n_head*head_dim] */
    const gguf_tensor_info_t *attn_k, *attn_v; /* [n_embd, n_kv*head_dim] */
    const gguf_tensor_info_t *attn_q_norm, *attn_k_norm; /* [head_dim] */
    const gguf_tensor_info_t *attn_out;        /* [n_head*head_dim, n_embd] */
    /* GDN */
    const gguf_tensor_info_t *wqkv;            /* [n_embd, conv_dim] */
    const gguf_tensor_info_t *wqkv_gate;       /* [n_embd, d_inner] */
    const gguf_tensor_info_t *ssm_conv1d;      /* [d_conv, conv_dim] F32 */
    const gguf_tensor_info_t *ssm_dt_bias;     /* [n_v_heads] F32 */
    const gguf_tensor_info_t *ssm_a;           /* [n_v_heads] F32 */
    const gguf_tensor_info_t *ssm_beta, *ssm_alpha; /* [n_embd, n_v_heads] */
    const gguf_tensor_info_t *ssm_norm;        /* [head_v_dim] F32 */
    const gguf_tensor_info_t *ssm_out;         /* [d_inner, n_embd] */
    /* états */
    float *kv_k, *kv_v;     /* [n_ctx, n_kv_head*head_dim] si attention */
    float *conv_state;      /* [conv_dim, d_conv] si GDN */
    float *ssm_state;       /* [n_v_heads, head_v_dim, head_k_dim] si GDN */
} qwen_layer_t;

typedef struct {
    gguf_context_t gguf;
    qwen_config_t  cfg;
    qwen_dims_t    dims;
    qwen_layer_t  *layers;      /* dims.n_layer entrées */
    const gguf_tensor_info_t *tok_embd;
    const gguf_tensor_info_t *output_norm;
    const gguf_tensor_info_t *output;         /* = tok_embd si tied */

    /* buffers d'activation (arène unique, alignée 64 o) */
    float *x;        /* résidu [n_embd] */
    float *xb;       /* sortie de norme [n_embd] */
    float *qkv;      /* GDN: [conv_dim] | attn: [2*n_head*head_dim] */
    float *kv_small; /* attn k puis v [2*n_kv*head_dim] */
    float *z;        /* gate GDN [d_inner] */
    float *ffn_buf;  /* silu(gate)*up [n_ff] */
    float *attn_out; /* [n_head*head_dim] */
    float *logits;   /* [vocab_size] */
    void  *arena;

    uint32_t n_ctx;  /* contexte alloué pour le KV-cache */
    uint32_t pos;    /* position courante (nb tokens déjà traités) */
} qwen_model_t;

/* Charge le modèle et alloue tout l'état. n_ctx = taille de contexte max
 * pour le KV-cache (les couches GDN sont O(1) quelle que soit la valeur).
 * Retourne 0 si succès, -1 sinon (ressources libérées). */
int  model_init(qwen_model_t *m, const char *path, uint32_t n_ctx);
void model_free(qwen_model_t *m);

#endif
