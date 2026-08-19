#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    char *text;
    float score;
    uint32_t id;
} token_entry_t;

typedef struct {
    uint32_t left_id;
    uint32_t right_id;
    uint32_t rank;
} bpe_merge_t;

typedef struct {
    uint32_t vocab_size;
    token_entry_t *vocab;
    uint32_t hash_mask;
    uint32_t *hash_table;

    uint32_t n_merges;
    bpe_merge_t *merges;
    uint32_t merge_hash_mask;
    uint32_t *merge_hash_table;

    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t unk_id;
    uint32_t pad_id;

    char byte_to_str[256][8];
    uint8_t cp_to_byte[0x160];
} tokenizer_t;

int tokenizer_init(tokenizer_t *tok, const gguf_context_t *gguf);
void tokenizer_free(tokenizer_t *tok);

uint32_t tokenizer_encode(const tokenizer_t *tok, const char *text, uint32_t *tokens, uint32_t max_tokens, bool add_bos);
const char *tokenizer_decode(const tokenizer_t *tok, uint32_t token_id);

#endif /* TOKENIZER_H */
