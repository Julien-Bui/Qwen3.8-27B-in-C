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

/* Ne jamais déréférencer un pointeur sur un champ de cette struct :
 * unpacked => accès non alignés possibles. Lecture via memcpy uniquement. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} __attribute__((packed)) gguf_header_t;

/* Type d'un TENSEUR (format de quantification ggml_type).
 * Valeurs vérifiées par arithmétique d'offsets sur le fichier réel. */
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
    char         name[GGUF_MAX_NAME]; /* tronqué à GGUF_MAX_NAME-1 caractères */
    uint32_t     n_dims;
    uint64_t     ne[4];               /* ne[0] = dim la plus interne (contiguë) */
    ggml_type_t  type;                /* format de quantification des poids */
    uint64_t     offset;              /* relatif à data_base, déjà aligné */
    const void  *data;                /* = data_base + offset, mappé PROT_READ */
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
    /* Architecture hybride (couches SSM + pleine attention entrelacées) */
    uint32_t full_attn_interval; /* 0 si pur transformeur */
    uint32_t ssm_state_size;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_inner_size;
    uint32_t ssm_group_count;
    uint32_t nextn_layers;       /* couches MTP (prédiction multi-token) */
} qwen_config_t;

typedef struct {
    int                 fd;           /* -1 si fermé */
    size_t              file_size;
    const void         *mmap_data;    /* mapping du fichier entier, PROT_READ */
    const void         *data_base;    /* début de la section de données */
    uint32_t            alignment;    /* propriété du fichier GGUF, pas du modèle */
    char                arch[32];     /* valeur de general.architecture ("qwen3") */
    gguf_header_t       header;
    qwen_config_t       config;       /* champs à 0 si clé absente */
    gguf_tensor_info_t *tensors;      /* tensor_count entrées */
    uint64_t            tensor_count;
} gguf_context_t;

/* gguf_open : retourne 0 en cas de succès, -1 en cas d'échec.
 * En cas d'échec : toutes les ressources sont libérées, ctx remis à zéro
 * (fd = -1), gguf_close(ctx) reste appelable sans danger.
 * En cas de succès : les pointeurs du ctx et tensors[].data sont valides
 * jusqu'à gguf_close(). */
int  gguf_open(gguf_context_t *ctx, const char *filepath);

/* gguf_close : libère mapping + tensors[]. Appelable sur un ctx zéro. */
void gguf_close(gguf_context_t *ctx);

/* Taille en octets d'une valeur du type donné (0 si taille variable). */
size_t gguf_type_size(gguf_type_t t);

/* Recherche par nom exact. Retourne NULL si absent. */
const gguf_tensor_info_t *gguf_find_tensor(const gguf_context_t *ctx, const char *name);

#endif
