/* Debug 2 cycles speculatifs vs reference sequentielle */
#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void run_cycle(qwen_model_t *m, pool_t *pool, tokenizer_t *tok,
                      const float *last_logits, uint32_t n_spec,
                      uint32_t *out_emitted, uint32_t *n_emit, uint32_t *out_j) {
    const uint32_t P = m->pos - 1;
    uint32_t cand[SPEC_MAX_B];
    cand[0] = model_sample_greedy(last_logits, m->cfg.vocab_size);
    for (uint32_t s = 1; s <= n_spec; s++) {
        mtp_forward(m, pool, cand[s - 1], P + s, 1);
        cand[s] = model_sample_greedy(m->logits, m->cfg.vocab_size);
    }
    model_forward_batch(m, pool, cand, P + 1, n_spec + 1);
    uint32_t j = n_spec;
    for (uint32_t i = 0; i < n_spec; i++) {
        uint32_t v = model_sample_greedy(m->logits_b + (size_t)i * m->cfg.vocab_size, m->cfg.vocab_size);
        if (v != cand[i + 1]) { j = i; break; }
    }
    printf("  cycle P=%u : cand = [%u \"%s\", %u \"%s\", %u \"%s\"] -> j=%u\n",
           P, cand[0], tokenizer_decode(tok, cand[0]),
           cand[1], tokenizer_decode(tok, cand[1]),
           cand[2], tokenizer_decode(tok, cand[2]), j);
    *n_emit = j + 1;
    *out_j = j;
    memcpy(out_emitted, cand, (j + 1) * sizeof(uint32_t));

    if (j < n_spec) {
        model_rollback_to(m, j);
        m->pos = P + j + 2;
    }
    memcpy(m->mtp_h, m->h_save + (size_t)j * m->dims.n_embd, m->dims.n_embd * sizeof(float));
    mtp_forward(m, pool, cand[j], P + j + 1, 0);
}

int main(int argc, char **argv) {
    qwen_model_t ms, mr;
    if (model_init(&ms, argv[1], 128) != 0) return 1;
    if (model_init(&mr, argv[1], 128) != 0) return 1;
    tokenizer_t tok;
    if (tokenizer_init(&tok, &ms.gguf) != 0) return 1;
    pool_t *pool = pool_create(12, 1);

    uint32_t ptoks[64];
    uint32_t n_prompt = tokenizer_encode(&tok, "The capital of France is", ptoks, 64, false);

    /* modele speculatif : prefill + ancre */
    for (uint32_t pos = 0; pos < n_prompt; pos++) {
        model_forward(&ms, pool, ptoks[pos], pos);
        memcpy(ms.mtp_h, ms.x, ms.dims.n_embd * sizeof(float));
        mtp_forward(&ms, pool, ptoks[pos], pos, 0);
    }

    /* modele reference : prefill seul */
    for (uint32_t pos = 0; pos < n_prompt; pos++)
        model_forward(&mr, pool, ptoks[pos], pos);

    uint32_t seq_stream[256];
    uint32_t n_seq = 0;

    const float *last_logits = ms.logits;
    for (int cycle = 0; cycle < 4; cycle++) {
        uint32_t ref_cand0 = model_sample_greedy(mr.logits, mr.cfg.vocab_size);
        char db0[64]; strncpy(db0, tokenizer_decode(&tok, ref_cand0), 63); db0[63] = 0;
        printf("ref cand0 = %u \"%s\"\n", ref_cand0, db0);

        uint32_t emitted[SPEC_MAX_B], n_emit, j;
        run_cycle(&ms, pool, &tok, last_logits, 2, emitted, &n_emit, &j);

        for (uint32_t i = 0; i < n_emit; i++) {
            uint32_t rg = model_sample_greedy(mr.logits, mr.cfg.vocab_size);
            char a[64], b2[64];
            strncpy(a, tokenizer_decode(&tok, emitted[i]), 63); a[63] = 0;
            strncpy(b2, tokenizer_decode(&tok, rg), 63); b2[63] = 0;
            if (rg != emitted[i])
                printf("  *** DIVERGENCE emis[%u]=%u \"%s\" mais ref greedy=%u \"%s\"\n",
                       i, emitted[i], a, rg, b2);
            model_forward(&mr, pool, emitted[i], mr.pos);
            seq_stream[n_seq++] = emitted[i];
        }
        last_logits = ms.logits_b + (size_t)j * ms.cfg.vocab_size;
    }

    pool_destroy(pool);
    model_free(&ms);
    model_free(&mr);
    tokenizer_free(&tok);
    return 0;
}
