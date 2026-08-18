#define _POSIX_C_SOURCE 200809L

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 16777619u;
    }
    return h;
}

static uint64_t pair_hash(uint32_t l, uint32_t r) {
    uint64_t key = ((uint64_t)l << 32) | (uint64_t)r;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

static int read_bytes(const uint8_t **p, const uint8_t *end, void *dst, size_t size) {
    if (size > (size_t)(end - *p)) return -1;
    memcpy(dst, *p, size);
    *p += size;
    return 0;
}

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
        for (uint64_t i = 0; i < count; i++)
            if (skip_value(p, end, (gguf_type_t)elem_type)) return -1;
        return 0;
    }
    default: {
        size_t sz = gguf_type_size(type);
        if (sz == 0 || sz > (size_t)(end - *p)) return -1;
        *p += sz;
        return 0;
    }
    }
}

static uint32_t token_find(const tokenizer_t *tok, const char *text) {
    uint32_t h = fnv1a(text) & tok->hash_mask;
    while (tok->hash_table[h] != UINT32_MAX) {
        uint32_t id = tok->hash_table[h];
        if (strcmp(tok->vocab[id].text, text) == 0) return id;
        h = (h + 1) & tok->hash_mask;
    }
    return UINT32_MAX;
}

static void token_insert(tokenizer_t *tok, uint32_t id) {
    uint32_t h = fnv1a(tok->vocab[id].text) & tok->hash_mask;
    while (tok->hash_table[h] != UINT32_MAX) {
        h = (h + 1) & tok->hash_mask;
    }
    tok->hash_table[h] = id;
}

static uint32_t merge_find(const tokenizer_t *tok, uint32_t l, uint32_t r) {
    uint64_t hk = pair_hash(l, r);
    uint32_t h = (uint32_t)(hk & tok->merge_hash_mask);
    while (tok->merge_hash_table[h] != UINT32_MAX) {
        uint32_t midx = tok->merge_hash_table[h];
        if (tok->merges[midx].left_id == l && tok->merges[midx].right_id == r)
            return tok->merges[midx].rank;
        h = (h + 1) & tok->merge_hash_mask;
    }
    return UINT32_MAX;
}

static void merge_insert(tokenizer_t *tok, uint32_t midx) {
    uint64_t hk = pair_hash(tok->merges[midx].left_id, tok->merges[midx].right_id);
    uint32_t h = (uint32_t)(hk & tok->merge_hash_mask);
    while (tok->merge_hash_table[h] != UINT32_MAX) {
        h = (h + 1) & tok->merge_hash_mask;
    }
    tok->merge_hash_table[h] = midx;
}

static void init_byte_table(tokenizer_t *tok) {
    int n = 0;
    memset(tok->cp_to_byte, 0xFF, sizeof tok->cp_to_byte);
    for (int b = 0; b < 256; b++) {
        uint32_t cp = 0;
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            cp = (uint32_t)b;
        } else {
            cp = 256 + n;
            n++;
        }
        tok->cp_to_byte[cp] = (uint8_t)b;
        if (cp < 0x80) {
            tok->byte_to_str[b][0] = (char)cp;
            tok->byte_to_str[b][1] = '\0';
        } else if (cp < 0x800) {
            tok->byte_to_str[b][0] = (char)(0xC0 | (cp >> 6));
            tok->byte_to_str[b][1] = (char)(0x80 | (cp & 0x3F));
            tok->byte_to_str[b][2] = '\0';
        }
    }
}

/* ================== Prétokenization (style GPT-2, approximation) ==================
 * Le schéma exact "qwen35" vit dans llama.cpp (vocab.cpp) ; cette version
 * implémente les catégories du regex GPT-2 :
 *   's|'t|'re|'ve|'m|'ll|'d | " "?L+ | " "?N+ | " "?other+ | \s+(?!\S) | \s+
 * Approximation unicode : toute séquence multioctet UTF-8 est assimilée à L.
 * Contrainte clé : AUCUNE fusion BPE ne doit traverser une frontière. */

typedef enum { CT_NONE = 0, CT_LETTER, CT_DIGIT, CT_OTHER, CT_SPACE } cat_t;

static cat_t byte_cat(uint8_t b) {
    if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) return CT_LETTER;
    if (b >= '0' && b <= '9') return CT_DIGIT;
    if (b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\f' || b == '\v') return CT_SPACE;
    if (b >= 0x80) return CT_LETTER;   /* approx \p{L} pour UTF-8 multioctet */
    return CT_OTHER;
}

/* Longueur d'une contraction type 's 't 're 've 'm 'll 'd, sinon 0 */
static size_t contraction_len(const char *s, size_t remain) {
    if (remain < 2 || s[0] != '\'') return 0;
    switch (s[1]) {
    case 's': case 't': case 'm': case 'd':
        return 2;
    case 'r': case 'v':
        return (remain >= 3 && s[2] == 'e') ? 3 : 0;
    case 'l':
        return (remain >= 3 && s[2] == 'l') ? 3 : 0;
    default:
        return 0;
    }
}

/* Découpe text en chunks [start, len). Retourne le nombre de chunks. */
static size_t pretokenize(const char *text, size_t len,
                          size_t *starts, size_t *lens, size_t max_chunks) {
    size_t i = 0, n = 0;
    while (i < len && n < max_chunks) {
        size_t clen = contraction_len(text + i, len - i);
        if (clen) {
            starts[n] = i; lens[n] = clen; n++;
            i += clen;
            continue;
        }
        if (byte_cat((uint8_t)text[i]) == CT_SPACE) {
            /* run de blancs : le DERNIER blanc se rattache au chunk suivant
             * s'il est suivi d'un non-blanc (sémantique \s+(?!\S)) */
            size_t j = i;
            while (j < len && byte_cat((uint8_t)text[j]) == CT_SPACE) j++;
            if (j == len) {                    /* blancs finaux : tout en un */
                starts[n] = i; lens[n] = j - i; n++;
                i = j;
                continue;
            }
            if (j - i > 1) {                   /* blancs internes + non-blanc */
                starts[n] = i; lens[n] = j - i - 1; n++;
                i = j - 1;                     /* le dernier blanc attend le chunk suivant */
            }
            /* un seul blanc : il devient la tête du chunk normal ci-dessous */
        }
        /* chunk normal : un blanc tête optionnel + run d'une catégorie */
        size_t start = i;
        size_t j = i;
        if (byte_cat((uint8_t)text[j]) == CT_SPACE) j++;   /* espace simple initial */
        if (j < len) {
            cat_t cat = byte_cat((uint8_t)text[j]);
            if (cat == CT_SPACE) {                        /* second blanc : chunk blanc seul */
                starts[n] = i; lens[n] = j - i; n++;
                i = j;
                continue;
            }
            while (j < len && byte_cat((uint8_t)text[j]) == cat) j++;
        }
        starts[n] = start; lens[n] = j - start; n++;
        i = j;
    }
    return n;
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

int tokenizer_init(tokenizer_t *tok, const gguf_context_t *gguf) {
    memset(tok, 0, sizeof *tok);
    if (!gguf || !gguf->mmap_data) return -1;

    init_byte_table(tok);

    const uint8_t *p = (const uint8_t *)gguf->mmap_data + sizeof(gguf_header_t);
    const uint8_t *end = (const uint8_t *)gguf->mmap_data + gguf->file_size;

    tok->bos_id = 151644;
    tok->eos_id = 151645;
    tok->unk_id = 151643;
    tok->pad_id = 151643;

    for (uint64_t i = 0; i < gguf->header.metadata_kv_count; i++) {
        char key[160];
        uint32_t vtype;
        if (read_string(&p, end, key, sizeof key)) return -1;
        if (read_bytes(&p, end, &vtype, sizeof vtype)) return -1;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            uint32_t elem_type;
            uint64_t count;
            if (read_bytes(&p, end, &elem_type, sizeof elem_type)) return -1;
            if (read_bytes(&p, end, &count, sizeof count)) return -1;
            tok->vocab_size = (uint32_t)count;
            tok->vocab = calloc(tok->vocab_size, sizeof(token_entry_t));
            tok->hash_mask = 0xFFFFF;
            tok->hash_table = malloc((tok->hash_mask + 1) * sizeof(uint32_t));
            memset(tok->hash_table, 0xFF, (tok->hash_mask + 1) * sizeof(uint32_t));

            for (uint32_t tid = 0; tid < tok->vocab_size; tid++) {
                uint64_t slen;
                if (read_bytes(&p, end, &slen, sizeof slen)) return -1;
                char *str = malloc(slen + 1);
                if (read_bytes(&p, end, str, slen)) { free(str); return -1; }
                str[slen] = '\0';
                tok->vocab[tid].text = str;
                tok->vocab[tid].id = tid;
                token_insert(tok, tid);
            }
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0) {
            uint32_t elem_type;
            uint64_t count;
            if (read_bytes(&p, end, &elem_type, sizeof elem_type)) return -1;
            if (read_bytes(&p, end, &count, sizeof count)) return -1;
            for (uint32_t tid = 0; tid < (uint32_t )count && tid < tok->vocab_size; tid++) {
                float sc = 0.0f;
                if (read_bytes(&p, end, &sc, sizeof sc)) return -1;
                tok->vocab[tid].score = sc;
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0) {
            uint32_t elem_type;
            uint64_t count;
            if (read_bytes(&p, end, &elem_type, sizeof elem_type)) return -1;
            if (read_bytes(&p, end, &count, sizeof count)) return -1;
            tok->n_merges = (uint32_t)count;
            tok->merges = calloc(tok->n_merges, sizeof(bpe_merge_t));
            tok->merge_hash_mask = 0xFFFFF;
            tok->merge_hash_table = malloc((tok->merge_hash_mask + 1) * sizeof(uint32_t));
            memset(tok->merge_hash_table, 0xFF, (tok->merge_hash_mask + 1) * sizeof(uint32_t));

            for (uint32_t midx = 0; midx < tok->n_merges; midx++) {
                uint64_t slen;
                if (read_bytes(&p, end, &slen, sizeof slen)) return -1;
                char mstr[256];
                size_t tc = (slen < sizeof(mstr) - 1) ? (size_t)slen : sizeof(mstr) - 1;
                if (read_bytes(&p, end, mstr, tc)) return -1;
                mstr[tc] = '\0';
                if (slen > tc) p += (slen - tc);

                char *sp = strchr(mstr, ' ');
                if (sp) {
                    *sp = '\0';
                    uint32_t lid = token_find(tok, mstr);
                    uint32_t rid = token_find(tok, sp + 1);
                    tok->merges[midx].left_id = lid;
                    tok->merges[midx].right_id = rid;
                    tok->merges[midx].rank = midx;
                    if (lid != UINT32_MAX && rid != UINT32_MAX) {
                        merge_insert(tok, midx);
                    }
                }
            }
        } else {
            skip_value(&p, end, (gguf_type_t)vtype);
        }
    }

    /* Tokens spéciaux : aucune clé GGUF d'ID -> résolution par texte.
     * Fallback : les IDs ChatML historiques de Qwen. */
    {
        uint32_t id;
        if ((id = token_find(tok, "<|im_end|>")) != UINT32_MAX) tok->eos_id = id;
        else if ((id = token_find(tok, "<|endoftext|>")) != UINT32_MAX) tok->eos_id = id;
        if ((id = token_find(tok, "<|im_start|>")) != UINT32_MAX) tok->bos_id = id;
        if ((id = token_find(tok, "<|UNUNK|>")) != UINT32_MAX) tok->unk_id = id;
        tok->pad_id = tok->eos_id;
    }

    return 0;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i].text);
        }
        free(tok->vocab);
    }
    free(tok->hash_table);
    free(tok->merges);
    free(tok->merge_hash_table);
    memset(tok, 0, sizeof *tok);
}

uint32_t tokenizer_encode(const tokenizer_t *tok, const char *text, uint32_t *tokens, uint32_t max_tokens, bool add_bos) {
    if (!tok || !text || !tokens || max_tokens == 0) return 0;
    uint32_t n = 0;
    if (add_bos && tok->bos_id != UINT32_MAX && n < max_tokens) {
        tokens[n++] = tok->bos_id;
    }

    size_t text_len = strlen(text);
    if (text_len == 0) return n;

    /* Prétokenization : les fusions BPE ne traversent JAMAIS un chunk */
    size_t *starts = malloc((text_len + 2) * sizeof(size_t));
    size_t *lens   = malloc((text_len + 2) * sizeof(size_t));
    uint32_t *symbols = malloc((text_len + 2) * sizeof(uint32_t));
    if (!starts || !lens || !symbols) {
        free(starts); free(lens); free(symbols);
        return n;
    }

    size_t n_chunks = pretokenize(text, text_len, starts, lens, text_len + 2);

    for (size_t c = 0; c < n_chunks && n < max_tokens; c++) {
        const char *chunk = text + starts[c];
        size_t chunk_len = lens[c];
        uint32_t n_symbols = 0;

        /* symboles initiaux : byte-level */
        for (size_t i = 0; i < chunk_len; i++) {
            uint8_t b = (uint8_t)chunk[i];
            uint32_t id = token_find(tok, tok->byte_to_str[b]);
            if (id == UINT32_MAX) {
                char byte_str[16];
                snprintf(byte_str, sizeof byte_str, "<0x%02X>", b);
                id = token_find(tok, byte_str);
                if (id == UINT32_MAX) id = tok->unk_id;
            }
            if (id == UINT32_MAX) continue;
            symbols[n_symbols++] = id;
        }

        /* BPE intra-chunk */
        while (n_symbols > 1) {
            uint32_t best_rank = UINT32_MAX;
            uint32_t best_idx = UINT32_MAX;

            for (uint32_t j = 0; j < n_symbols - 1; j++) {
                uint32_t r = merge_find(tok, symbols[j], symbols[j + 1]);
                if (r < best_rank) {
                    best_rank = r;
                    best_idx = j;
                }
            }

            if (best_rank == UINT32_MAX) break;

            const char *s1 = tok->vocab[symbols[best_idx]].text;
            const char *s2 = tok->vocab[symbols[best_idx + 1]].text;
            size_t l1 = strlen(s1), l2 = strlen(s2);
            char comb[512];
            if (l1 + l2 >= sizeof comb) break;
            memcpy(comb, s1, l1);
            memcpy(comb + l1, s2, l2);
            comb[l1 + l2] = '\0';
            uint32_t merged_id = token_find(tok, comb);
            if (merged_id == UINT32_MAX) break;

            symbols[best_idx] = merged_id;
            for (uint32_t j = best_idx + 1; j < n_symbols - 1; j++) {
                symbols[j] = symbols[j + 1];
            }
            n_symbols--;
        }

        for (uint32_t i = 0; i < n_symbols && n < max_tokens; i++) {
            tokens[n++] = symbols[i];
        }
    }

    free(starts);
    free(lens);
    free(symbols);
    return n;
}

static char decode_buf[512];

/* Décodage byte-level complet : UTF-8 -> cp -> octet brut via cp_to_byte.
 * Les cps non mappés (textes unicode réels) passent tels quels. */
const char *tokenizer_decode(const tokenizer_t *tok, uint32_t token_id) {
    if (!tok || token_id >= tok->vocab_size || !tok->vocab[token_id].text) return "";
    const char *src = tok->vocab[token_id].text;
    size_t dst_i = 0;
    for (size_t i = 0; src[i] != '\0' && dst_i + 4 < sizeof(decode_buf); ) {
        uint8_t b0 = (uint8_t)src[i];
        uint32_t cp;
        size_t adv;
        if (b0 < 0x80) {
            cp = b0;
            adv = 1;
        } else if ((b0 & 0xE0) == 0xC0 && (uint8_t)src[i + 1] != '\0') {
            cp = ((uint32_t)(b0 & 0x1F) << 6) | ((uint8_t)src[i + 1] & 0x3F);
            adv = 2;
        } else {
            /* 3-4 octets : unicode direct, pas dans la table byte-level */
            decode_buf[dst_i++] = src[i++];
            continue;
        }
        uint8_t raw = (cp < sizeof tok->cp_to_byte) ? tok->cp_to_byte[cp] : 0xFF;
        if (raw != 0xFF) {
            decode_buf[dst_i++] = (char)raw;
            i += adv;
        } else {
            /* cp non mappé : recopier les octets originaux */
            for (size_t k = 0; k < adv && src[i + k]; k++)
                decode_buf[dst_i++] = src[i + k];
            i += adv;
        }
    }
    decode_buf[dst_i] = '\0';
    return decode_buf;
}
