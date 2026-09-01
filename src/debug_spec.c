/* Debug cycle speculatif : trace cand/j/g par cycle */
#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    qwen_model_t model;
    if (model_init(&model, argv[1], 128) != 0) return 1;
    tokenizer_t tok;
    if (tokenizer_init(&tok, &model.gguf) != 0) return 1;
    pool_t *pool = pool_create(12, 1);

    const char *prompt = "The capital of France is";
    uint32_t ptoks[64];
    uint32_t n_prompt = tokenizer_encode(&tok, prompt, ptoks, 64, false);
    printf("prompt tokens (%u):", n_prompt);
    for (uint32_t i = 0; i < n_prompt; i++) printf(" %u", ptoks[i]);
    printf("\n");

    /* prefill + ancre MTP */
    for (uint32_t pos = 0; pos < n_prompt; pos++) {
        model_forward(&model, pool, ptoks[pos], pos);
        memcpy(model.mtp_h, model.x, model.dims.n_embd * sizeof(float));
        mtp_forward(&model, pool, ptoks[pos], pos, 0);
    }

    uint32_t greedy0 = model_sample_greedy(model.logits, model.cfg.vocab_size);
    printf("argmax trunk (apres prompt) = %u \"%s\"\n", greedy0, tokenizer_decode(&tok, greedy0));

    /* drafts */
    const uint32_t P = model.pos - 1;
    uint32_t cand[SPEC_MAX_B];
    cand[0] = greedy0;
    for (uint32_t s = 1; s <= 2; s++) {
        mtp_forward(&model, pool, cand[s - 1], P + s, 1);
        cand[s] = model_sample_greedy(model.logits, model.cfg.vocab_size);
        printf("draft %u : MTP(t=%u, pos=%u) -> %u \"%s\"\n",
               s, cand[s - 1], P + s, cand[s], tokenizer_decode(&tok, cand[s]));
    }

    /* verification batch */
    model_forward_batch(&model, pool, cand, P + 1, 3, 1);
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t v = model_sample_greedy(model.logits_b + (size_t)i * model.cfg.vocab_size,
                                         model.cfg.vocab_size);
        printf("verify logits_b[%u] argmax = %u \"%s\"  (cand[%u]=%u)\n",
               i, v, tokenizer_decode(&tok, v), i, cand[i]);
    }

    /* reference : forwards sequentiels des memes candidats */
    qwen_model_t m2;
    if (model_init(&m2, argv[1], 128) != 0) return 1;
    for (uint32_t pos = 0; pos < n_prompt; pos++) model_forward(&m2, pool, ptoks[pos], pos);
    for (uint32_t i = 0; i < 3; i++) {
        model_forward(&m2, pool, cand[i], n_prompt + i);
        uint32_t v = model_sample_greedy(m2.logits, m2.cfg.vocab_size);
        printf("seq forward(cand[%u]) argmax = %u \"%s\"\n", i, v, tokenizer_decode(&tok, v));
    }

    pool_destroy(pool);
    model_free(&model);
    model_free(&m2);
    tokenizer_free(&tok);
    return 0;
}
