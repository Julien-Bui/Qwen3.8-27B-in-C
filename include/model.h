#ifndef MODEL_H
#define MODEL_H

#include "gguf.h"
#include "threadpool.h"
#include <stdint.h>
#include <stdbool.h>

/* Dimensions derived from GGUF metadata */
typedef struct {
    /* Common */
    uint32_t n_embd;        /* 5120 */
    uint32_t n_ff;          /* 17408 */
    uint32_t n_layer;       /* 64 backbone layers (MTP excluded) */
    uint32_t n_attn_layer;  /* 16 Full Attention layers */
    uint32_t n_recr_layer;  /* 48 Gated DeltaNet layers */
    /* Full Attention (Gated GQA) */
    uint32_t n_head;        /* 24 */
    uint32_t n_kv_head;     /* 4 */
    uint32_t head_dim;      /* 256 */
    uint32_t rope_sections[4];
    /* Gated DeltaNet (SSM) */
    uint32_t n_v_heads;     /* 48 (= dt_rank) */
    uint32_t n_k_heads;     /* 16 (= group_count) */
    uint32_t head_k_dim;    /* 128 */
    uint32_t head_v_dim;    /* 128 */
    uint32_t d_inner;       /* 6144 */
    uint32_t conv_dim;      /* 10240 */
    uint32_t d_conv;        /* 4 */
    uint32_t d_state;       /* 128 */
} qwen_dims_t;

typedef struct {
    bool is_recurrent;      /* true = GDN, false = Full Attention */
    /* Shared */
    const gguf_tensor_info_t *attn_norm;       /* Pre-attention RMSNorm */
    const gguf_tensor_info_t *post_attn_norm;  /* Pre-FFN RMSNorm */
    const gguf_tensor_info_t *ffn_gate, *ffn_up, *ffn_down;
    /* Full Attention */
    const gguf_tensor_info_t *attn_q;          /* [n_embd, 2*n_head*head_dim] */
    const gguf_tensor_info_t *attn_k, *attn_v; /* [n_embd, n_kv*head_dim] */
    const gguf_tensor_info_t *attn_q_norm, *attn_k_norm; /* [head_dim] */
    const gguf_tensor_info_t *attn_out;       /* [n_head*head_dim, n_embd] */
    /* GDN */
    const gguf_tensor_info_t *wqkv;           /* [n_embd, conv_dim] */
    const gguf_tensor_info_t *wqkv_gate;      /* [n_embd, d_inner] */
    const gguf_tensor_info_t *ssm_conv1d;      /* [d_conv, conv_dim] F32 */
    const gguf_tensor_info_t *ssm_dt_bias;    /* [n_v_heads] F32 */
    const gguf_tensor_info_t *ssm_a;           /* [n_v_heads] F32 */
    const gguf_tensor_info_t *ssm_beta, *ssm_alpha; /* [n_embd, n_v_heads] */
    const gguf_tensor_info_t *ssm_norm;        /* [head_v_dim] F32 */
    const gguf_tensor_info_t *ssm_out;         /* [d_inner, n_embd] */
    /* States */
    float *kv_k, *kv_v;    /* [n_ctx, n_kv_head*head_dim] if attention */
    float *conv_state;     /* [conv_dim, d_conv] if GDN */
    float *ssm_state;      /* [n_v_heads, head_v_dim, head_k_dim] if GDN */
} qwen_layer_t;

typedef struct {
    gguf_context_t gguf;
    qwen_config_t  cfg;
    qwen_dims_t    dims;
    qwen_layer_t  *layers;
    const gguf_tensor_info_t *tok_embd;
    const gguf_tensor_info_t *output_norm;
    const gguf_tensor_info_t *output;

    /* Final hidden state cached for MTP layer */
    float *mtp_h;

    /* Activation buffers (single arena, 64-byte aligned) */
    float *x;       /* Residual [n_embd] */
    float *xb;      /* Norm output [n_embd] */
    float *qkv;     /* GDN: [conv_dim] | attn: [2*n_head*head_dim] */
    float *kv_small; /* Attn k then v [2*n_kv*head_dim] */
    float *z;       /* GDN gate [d_inner] */
    float *ffn_buf;  /* silu(gate)*up [n_ff] */
    float *attn_out; /* [n_head*head_dim] */
    float *logits;   /* [vocab_size] */
    float *scores;   /* [n_ctx] for attention softmax */

    /* Precomputed RoPE tables: [n_ctx][rope_dims/2] */
    float *rope_cos, *rope_sin;
    void  *arena;

    /* Speculative decoding batch buffers: [SPEC_MAX_B][dim] */
    float *bx;          /* Residual */
    float *bxb;         /* Norm output */
    float *bqkv;        /* GDN: conv_dim | attn: 2*n_head*head_dim */
    float *bkv;         /* Attn k then v */
    float *bz;          /* GDN gate */
    float *bffn;        /* silu(gate)*up */
    float *battn_out;   /* Head output */
    float *h_save;      /* Saved hidden state per token for MTP */
    float *logits_b;    /* [SPEC_MAX_B][vocab_size] */
    void  *arena_b;

    /* Recurrent state checkpoints for rollbacks */
    float *ckpt_conv;
    float *ckpt_ssm;

    /* MTP Layer (blk.64.* = nextn) */
    qwen_layer_t mtp_layer;
    const gguf_tensor_info_t *mtp_eh_proj;     /* [10240, 5120] Q8_0 */
    const gguf_tensor_info_t *mtp_enorm;       /* [5120] */
    const gguf_tensor_info_t *mtp_hnorm;       /* [5120] */
    const gguf_tensor_info_t *mtp_shnorm;      /* [5120] shared_head_norm */

    uint32_t n_ctx;
    uint32_t pos;
} qwen_model_t;

#define SPEC_MAX_B 3     /* k = SPEC_MAX_B - 1 drafts (speculative verification) */
#define PREFILL_MAX_B 8  /* max tokens per batched prefill chunk */

/* Model initialization and destruction */
int  model_init(qwen_model_t *m, const char *path, uint32_t n_ctx);
void model_free(qwen_model_t *m);

/* Computation layers */
void attention_layer(qwen_model_t *m, qwen_layer_t *L, pool_t *pool, uint32_t pos);
void gdn_layer(qwen_model_t *m, qwen_layer_t *L, pool_t *pool);

/* Full single-token forward pass */
void model_forward(qwen_model_t *m, pool_t *pool, uint32_t token, uint32_t pos);

/* Batched forward pass (speculative verification or prefill chunks).
 * ckpt: save per-token recurrent state checkpoints (needed for rollback,
 *       i.e. speculative verification; set 0 for prefill). */
void model_forward_batch(qwen_model_t *m, pool_t *pool,
                         const uint32_t *tokens, uint32_t pos, uint32_t B, int ckpt);

/* Recurrent state snapshot and rollback */
int  model_snapshot_states(qwen_model_t *m);
void model_rollback_states(qwen_model_t *m);
void model_rollback_to(qwen_model_t *m, uint32_t b);

/* MTP Multi-Token Prediction layer forward pass */
int mtp_forward(qwen_model_t *m, pool_t *pool, uint32_t token, uint32_t pos,
                int want_logits);

/* Batched MTP forward pass (prefill): fills the MTP KV cache for
 * tokens[pos..pos+B-1]; mtp_h anchors on the last token's output. */
int mtp_forward_batch(qwen_model_t *m, pool_t *pool, const uint32_t *tokens,
                      uint32_t pos, uint32_t B, int want_logits);

/* Greedy argmax */
uint32_t model_sample_greedy(const float *logits, uint32_t vocab_size);

/* Profiling counters */
extern double qwen_prof_embed, qwen_prof_gdn, qwen_prof_attn, qwen_prof_head;
extern uint64_t qwen_prof_calls;

#endif /* MODEL_H */
