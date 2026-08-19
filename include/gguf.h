#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGUF_MAGIC 0x46554747
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_MAX_NAME 64

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

/* Packed GGUF header (read via memcpy to avoid unaligned access) */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} __attribute__((packed)) gguf_header_t;

/* Tensor quantization format type (matching GGML type indices) */
typedef enum {
    GGML_F32     = 0,
    GGML_F16     = 1,
    GGML_Q4_0    = 2,
    GGML_Q4_1    = 3,
    GGML_Q5_0    = 6,
    GGML_Q5_1    = 7,
    GGML_Q8_0    = 8,
    GGML_Q2_K    = 10,
    GGML_Q3_K    = 11,
    GGML_Q4_K    = 12,
    GGML_Q5_K    = 13,
    GGML_Q6_K    = 14,
    GGML_Q8_K    = 15,
    GGML_IQ2_XXS = 16,
    GGML_IQ2_XS  = 17,
    GGML_IQ3_XXS = 18,
    GGML_IQ1_S   = 19,
    GGML_IQ4_NL  = 20,
    GGML_IQ3_S   = 21,
    GGML_IQ2_S   = 22,
    GGML_IQ4_XS  = 23,
    GGML_IQ1_M   = 24,
    GGML_BF16    = 25,
} ggml_type_t;

typedef struct {
    char         name[GGUF_MAX_NAME]; /* Tensor name */
    uint32_t     n_dims;
    uint64_t     ne[4];               /* ne[0] = innermost contiguous dimension */
    ggml_type_t  type;                /* Weight quantization format */
    uint64_t     offset;              /* Relative to data_base, aligned */
    const void  *data;                /* = data_base + offset, mapped PROT_READ */
} gguf_tensor_info_t;

typedef struct {
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;          /* attention.key_length */
    uint32_t n_layer;
    uint32_t n_ff;
    uint32_t vocab_size;
    uint32_t context_len;
    float    rope_theta;
    uint32_t rope_dim_count;    /* rope.dimension_count */
    uint32_t rope_sections[4];  /* rope.dimension_sections (mrope) */
    float    rms_norm_eps;
    bool     tie_word_embeddings;
    /* Hybrid architecture (interleaved SSM and full attention layers) */
    uint32_t full_attn_interval;
    uint32_t ssm_state_size;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_inner_size;
    uint32_t ssm_group_count;
    uint32_t nextn_layers;       /* MTP layers (multi-token prediction) */
} qwen_config_t;

typedef struct {
    int                 fd;
    size_t              file_size;
    const void         *mmap_data;    /* Whole file mmap, PROT_READ */
    const void         *data_base;    /* Start of tensor binary data */
    uint32_t            alignment;
    char                arch[32];     /* general.architecture ("qwen3") */
    gguf_header_t       header;
    qwen_config_t       config;
    gguf_tensor_info_t *tensors;      /* tensor_count entries */
    uint64_t            tensor_count;
} gguf_context_t;

/* Opens and parses GGUF file. Returns 0 on success, -1 on failure. */
int  gguf_open(gguf_context_t *ctx, const char *filepath);

/* Closes file and unmaps memory. */
void gguf_close(gguf_context_t *ctx);

/* Returns byte size of a GGUF metadata type (0 if variable). */
size_t gguf_type_size(gguf_type_t t);

/* Finds tensor by exact name. Returns NULL if absent. */
const gguf_tensor_info_t *gguf_find_tensor(const gguf_context_t *ctx, const char *name);

#endif /* GGUF_H */
