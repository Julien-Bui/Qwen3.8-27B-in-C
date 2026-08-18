#define _POSIX_C_SOURCE 200809L

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <inttypes.h>

/* ---------- Taille des types GGUF (0 = taille variable) ---------- */
size_t gguf_type_size(gguf_type_t t) {
    switch (t) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:    return 1;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:   return 2;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32: return 4;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64: return 8;
    default:                return 0; /* STRING, ARRAY */
    }
}

/* ---------- Lectures curseur : lisent, vérifient les bornes, avancent ---------- */

/* Critère de débordement SÛR : (end - *p) ne peut pas déborder,
 * contrairement à (*p + size) qui peut boucler sur des size énormes. */
static int read_bytes(const uint8_t **p, const uint8_t *end, void *dst, size_t size) {
    if (size > (size_t)(end - *p)) return -1;
    memcpy(dst, *p, size);
    *p += size;
    return 0;
}

static int read_string(const uint8_t **p, const uint8_t *end, char *dst, size_t max_len) {
    uint64_t len;
    if (read_bytes(p, end, &len, sizeof len)) return -1;
    if (len > (uint64_t)(end - *p)) return -1;

    size_t to_copy = (len < max_len - 1) ? (size_t)len : max_len - 1;
    memcpy(dst, *p, to_copy);
    dst[to_copy] = '\0';
    *p += len;
    return 0;
}

/* Saute une valeur de type quelconque (récursif pour les arrays). */
static int skip_value(const uint8_t **p, const uint8_t *end, gguf_type_t type) {
    switch (type) {
    case GGUF_TYPE_STRING: {
        uint64_t len;
        if (read_bytes(p, end, &len, sizeof len)) return -1;
        if (len > (uint64_t)(end - *p)) return -1;
        *p += len;
        return 0;
    }
    case GGUF_TYPE_ARRAY: {
        uint32_t elem_type;
        uint64_t count;
        if (read_bytes(p, end, &elem_type, sizeof elem_type)) return -1;
        if (read_bytes(p, end, &count, sizeof count)) return -1;
        if (elem_type > GGUF_TYPE_FLOAT64) return -1;
        for (uint64_t i = 0; i < count; i++)
            if (skip_value(p, end, (gguf_type_t)elem_type)) return -1;
        return 0;
    }
    default: { /* type à taille fixe */
        uint8_t scratch[8];
        return read_bytes(p, end, scratch, gguf_type_size(type));
    }
    }
}

/* ---------- Métadonnées ----------
 * pass 1 : ne capte que general.architecture et general.alignment.
 * pass 2 : rejoue le curseur et capte les clés "<arch>.*" dans la config. */
static int parse_metadata(gguf_context_t *ctx, const uint8_t **p, const uint8_t *end, int pass) {
    /* Table pilotée : suffixe de clé + type attendu + destination.
     * Vérifiez les noms exacts avec `gguf-dump.py` — ce sont ceux de llama.cpp. */
    const struct {
        const char  *suffix;
        gguf_type_t  type;
        void        *dst;
    } fields[] = {
        { ".embedding_length",                 GGUF_TYPE_UINT32,  &ctx->config.n_embd },
        { ".attention.head_count",             GGUF_TYPE_UINT32,  &ctx->config.n_head },
        { ".attention.head_count_kv",          GGUF_TYPE_UINT32,  &ctx->config.n_head_kv },
        { ".attention.key_length",             GGUF_TYPE_UINT32,  &ctx->config.head_dim },
        { ".block_count",                      GGUF_TYPE_UINT32,  &ctx->config.n_layer },
        { ".feed_forward_length",              GGUF_TYPE_UINT32,  &ctx->config.n_ff },
        { ".vocab_size",                       GGUF_TYPE_UINT32,  &ctx->config.vocab_size },
        { ".context_length",                   GGUF_TYPE_UINT32,  &ctx->config.context_len },
        { ".rope.freq_base",                   GGUF_TYPE_FLOAT32, &ctx->config.rope_theta },
        { ".rope.dimension_count",             GGUF_TYPE_UINT32,  &ctx->config.rope_dim_count },
        { ".attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, &ctx->config.rms_norm_eps },
        { ".tie_word_embeddings",              GGUF_TYPE_BOOL,    &ctx->config.tie_word_embeddings },
        { ".full_attention_interval",          GGUF_TYPE_UINT32,  &ctx->config.full_attn_interval },
        { ".ssm.state_size",                   GGUF_TYPE_UINT32,  &ctx->config.ssm_state_size },
        { ".ssm.conv_kernel",                  GGUF_TYPE_UINT32,  &ctx->config.ssm_conv_kernel },
        { ".ssm.inner_size",                   GGUF_TYPE_UINT32,  &ctx->config.ssm_inner_size },
        { ".ssm.group_count",                  GGUF_TYPE_UINT32,  &ctx->config.ssm_group_count },
        { ".nextn_predict_layers",             GGUF_TYPE_UINT32,  &ctx->config.nextn_layers },
    };
    const size_t n_fields = sizeof fields / sizeof fields[0];

    for (uint64_t i = 0; i < ctx->header.metadata_kv_count; i++) {
        char    key[160];
        uint32_t vtype;

        if (read_string(p, end, key, sizeof key)) return -1;
        if (read_bytes(p, end, &vtype, sizeof vtype)) return -1;
        if (vtype > GGUF_TYPE_FLOAT64) return -1;
        gguf_type_t t = (gguf_type_t)vtype;

        if (pass == 1) {
            if (strcmp(key, "general.architecture") == 0 && t == GGUF_TYPE_STRING) {
                if (read_string(p, end, ctx->arch, sizeof ctx->arch)) return -1;
                continue;
            }
            if (strcmp(key, "general.alignment") == 0 && t == GGUF_TYPE_UINT32) {
                uint32_t a;
                if (read_bytes(p, end, &a, sizeof a)) return -1;
                if (a > 0) ctx->alignment = a;
                continue;
            }
            if (skip_value(p, end, t)) return -1;
        } else {
            /* Cas spécial : rope.dimension_sections = array de 4 uint32 */
            char expect_secs[192];
            snprintf(expect_secs, sizeof expect_secs, "%s.rope.dimension_sections", ctx->arch);
            if (strcmp(key, expect_secs) == 0 && t == GGUF_TYPE_ARRAY) {
                uint32_t elem_type;
                uint64_t count;
                if (read_bytes(p, end, &elem_type, sizeof elem_type)) return -1;
                if (read_bytes(p, end, &count, sizeof count)) return -1;
                if (elem_type != GGUF_TYPE_UINT32 && elem_type != GGUF_TYPE_INT32) return -1;
                for (uint64_t k = 0; k < count; k++) {
                    uint32_t v;
                    if (read_bytes(p, end, &v, sizeof v)) return -1;
                    if (k < 4) ctx->config.rope_sections[k] = v;
                }
                continue;
            }
            int consumed = 0;
            for (size_t j = 0; j < n_fields && !consumed; j++) {
                char expect[192];
                snprintf(expect, sizeof expect, "%s%s", ctx->arch, fields[j].suffix);
                if (strcmp(key, expect) == 0 && t == fields[j].type) {
                    if (t == GGUF_TYPE_BOOL) {
                        uint8_t b;
                        if (read_bytes(p, end, &b, 1)) return -1;
                        *(bool *)fields[j].dst = (b != 0);
                    } else {
                        if (read_bytes(p, end, fields[j].dst, gguf_type_size(t))) return -1;
                    }
                    consumed = 1;
                }
            }
            if (!consumed && skip_value(p, end, t)) return -1;
        }
    }
    return 0;
}

/* ---------- Ouverture / fermeture ---------- */

void gguf_close(gguf_context_t *ctx) {
    if (!ctx) return;
    if (ctx->mmap_data && ctx->mmap_data != MAP_FAILED)
        munmap((void *)ctx->mmap_data, ctx->file_size);
    if (ctx->fd >= 0)
        close(ctx->fd);
    free(ctx->tensors);
    memset(ctx, 0, sizeof *ctx);
    ctx->fd = -1;
}

int gguf_open(gguf_context_t *ctx, const char *filepath) {
    memset(ctx, 0, sizeof *ctx);
    ctx->fd = -1;
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    /* 1. open + fstat */
    ctx->fd = open(filepath, O_RDONLY);
    if (ctx->fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(ctx->fd, &st) != 0 || st.st_size < (off_t)sizeof(gguf_header_t)) {
        fprintf(stderr, "gguf_open: fichier trop petit ou fstat échoué\n");
        goto fail;
    }
    ctx->file_size = (size_t)st.st_size;

    /* 2. mmap : attention, échec == MAP_FAILED, pas NULL */
    ctx->mmap_data = mmap(NULL, ctx->file_size, PROT_READ, MAP_SHARED, ctx->fd, 0);
    if (ctx->mmap_data == MAP_FAILED) {
        ctx->mmap_data = NULL;
        perror("mmap");
        goto fail;
    }

    const uint8_t *p   = ctx->mmap_data;
    const uint8_t *end = p + ctx->file_size;

    /* 3. Header (packed => jamais de cast direct, on passe par read_bytes/memcpy) */
    if (read_bytes(&p, end, &ctx->header, sizeof ctx->header)) goto fail;
    if (ctx->header.magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf_open: magic invalide\n");
        goto fail;
    }
    if (ctx->header.version < 2 || ctx->header.version > 3) {
        fprintf(stderr, "gguf_open: version %u non supportée\n", ctx->header.version);
        goto fail;
    }

    /* 4. Métadonnées en deux passes (l'ordre des clés dans le fichier est arbitraire) */
    const uint8_t *kv_start = p;
    if (parse_metadata(ctx, &p, end, 1)) {
        fprintf(stderr, "gguf_open: métadonnées invalides (passe 1)\n");
        goto fail;
    }
    if (ctx->arch[0] == '\0') {
        fprintf(stderr, "gguf_open: general.architecture absent\n");
        goto fail;
    }
    p = kv_start;
    if (parse_metadata(ctx, &p, end, 2)) {
        fprintf(stderr, "gguf_open: métadonnées invalides (passe 2)\n");
        goto fail;
    }

    /* 5. Infos des tenseurs */
    if (ctx->header.tensor_count == 0 || ctx->header.tensor_count > 100000) {
        fprintf(stderr, "gguf_open: tensor_count suspect (%llu)\n",
                (unsigned long long)ctx->header.tensor_count);
        goto fail;
    }
    ctx->tensor_count = ctx->header.tensor_count;
    ctx->tensors = calloc(ctx->tensor_count, sizeof *ctx->tensors);
    if (!ctx->tensors) goto fail;

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        gguf_tensor_info_t *t = &ctx->tensors[i];
        uint32_t type;

        if (read_string(&p, end, t->name, sizeof t->name)) {
            fprintf(stderr, "gguf_open: lecture nom tenseur #%" PRIu64 " échouée\n", i);
            goto fail;
        }
        if (read_bytes(&p, end, &t->n_dims, sizeof t->n_dims)) {
            fprintf(stderr, "gguf_open: lecture n_dims '%s' échouée\n", t->name);
            goto fail;
        }
        if (t->n_dims > 4) {
            fprintf(stderr, "gguf_open: n_dims invalide (%u) pour '%s'\n", t->n_dims, t->name);
            goto fail;
        }
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (read_bytes(&p, end, &t->ne[d], sizeof t->ne[d])) {
                fprintf(stderr, "gguf_open: lecture ne[%u] '%s' échouée\n", d, t->name);
                goto fail;
            }
        }
        if (read_bytes(&p, end, &type, sizeof type)) {
            fprintf(stderr, "gguf_open: lecture type '%s' échouée\n", t->name);
            goto fail;
        }
        t->type = (ggml_type_t)type;
        if (read_bytes(&p, end, &t->offset, sizeof t->offset)) {
            fprintf(stderr, "gguf_open: lecture offset '%s' échouée\n", t->name);
            goto fail;
        }
    }

    /* 6. Base de la section de données : position courante arrondie à l'alignement */
    uint64_t cur  = (uint64_t)(p - (const uint8_t *)ctx->mmap_data);
    uint64_t base = ((cur + ctx->alignment - 1) / ctx->alignment) * ctx->alignment;
    if (base >= ctx->file_size) goto fail;
    ctx->data_base = (const uint8_t *)ctx->mmap_data + base;

    /* 7. Résolution des pointeurs + garde minimale */
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        gguf_tensor_info_t *t = &ctx->tensors[i];
        t->data = (const uint8_t *)ctx->data_base + t->offset;
        if ((const uint8_t *)t->data >= end) {
            fprintf(stderr, "gguf_open: offset hors fichier pour '%s'\n", t->name);
            goto fail;
        }
    }
    /* 8. Fallback vocab : déduit de token_embd si la clé est absente */
    const gguf_tensor_info_t *embd = gguf_find_tensor(ctx, "token_embd.weight");
    if (embd && ctx->config.vocab_size == 0 && embd->n_dims == 2)
        ctx->config.vocab_size = (uint32_t)embd->ne[1];
    return 0;

fail:
    gguf_close(ctx);
    return -1;
}

const gguf_tensor_info_t *gguf_find_tensor(const gguf_context_t *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->tensor_count; i++)
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    return NULL;
}
