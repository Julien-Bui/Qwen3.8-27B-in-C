/* Test isole : batch -> rollback -> batch  vs  sequentiel */
#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    qwen_model_t mx, my;
    if (model_init(&mx, argv[1], 128) != 0) return 1;
    if (model_init(&my, argv[1], 128) != 0) return 1;
    tokenizer_t tok;
    if (tokenizer_init(&tok, &mx.gguf) != 0) return 1;
    pool_t *pool = pool_create(12, 1);

    uint32_t ptoks[64];
    uint32_t n_prompt = tokenizer_encode(&tok, "The capital of France is", ptoks, 64, false);
    for (uint32_t i = 0; i < n_prompt; i++) {
        model_forward(&mx, pool, ptoks[i], i);
        model_forward(&my, pool, ptoks[i], i);
    }

    /* batch 1 : [" Paris", ".", "<|im_end|>] @ 5..7, accepte j=1 */
    uint32_t cand1[3] = {11751, 13, 248046};
    model_forward_batch(&mx, pool, cand1, 5, 3, 1);
    model_rollback_to(&mx, 1);
    mx.pos = 5 + 1 + 2;   /* = 7 */

    /* reference : les 2 tokens acceptes en sequentiel */
    model_forward(&my, pool, 11751, 5);
    model_forward(&my, pool, 13, 6);

    /* batch 2 sur l'etat rollbacke : ["\n", "The", " capital"] @ 7..9 */
    uint32_t cand2[3] = {198, 760, 6511};
    model_forward_batch(&mx, pool, cand2, 7, 3, 1);

    /* reference : logits sauvegardes apres CHAQUE token */
    static float ref_logits[2][248320];
    for (uint32_t i = 0; i < 2; i++) {
        model_forward(&my, pool, cand2[i], 7 + i);
        memcpy(ref_logits[i], my.logits, 248320 * sizeof(float));
    }

    for (uint32_t b = 0; b < 2; b++) {
        const float *lb = mx.logits_b + (size_t)b * 248320;
        double maxd = 0;
        for (uint32_t v = 0; v < 248320; v += 13) {
            double d = fabs((double)lb[v] - ref_logits[b][v]);
            if (d > maxd) maxd = d;
        }
        uint32_t gx = model_sample_greedy(lb, 248320);
        uint32_t gy = model_sample_greedy(ref_logits[b], 248320);
        printf("batch2[%u] : maxdiff=%.3e  argmax x=%u y=%u %s\n",
               b, maxd, gx, gy, gx == gy ? "OK" : "*** DIVERGE ***");
    }

    pool_destroy(pool);
    model_free(&mx);
    model_free(&my);
    tokenizer_free(&tok);
    return 0;
}
