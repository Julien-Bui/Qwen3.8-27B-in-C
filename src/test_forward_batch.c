/* Test model_forward_batch: verify batched logits == sequential logits */
#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    qwen_model_t ma, mb;
    if (model_init(&ma, argv[1], 128) != 0) { fprintf(stderr, "init ma KO\n"); return 1; }
    if (model_init(&mb, argv[1], 128) != 0) { fprintf(stderr, "init mb KO\n"); return 1; }
    pool_t *pool = pool_create(12, 1);

    uint32_t toks[8];
    toks[0] = 151644;  /* <|im_start|> */
    toks[1] = 872;     /* user */
    toks[2] = 198;
    toks[3] = 9419;    /* Hello */
    toks[4] = 11;      /* , */
    toks[5] = 1204;    /* how */
    toks[6] = 513;     /* are */
    toks[7] = 488;     /* you */
    uint32_t B = 3;
    if (argc > 2) B = (uint32_t)atoi(argv[2]);
    if (B > SPEC_MAX_B) B = SPEC_MAX_B;

    /* passe sequentielle en sauvegardant les logits a chaque pas */
    static float seq_logits[SPEC_MAX_B][248320];
    for (uint32_t i = 0; i < B; i++) {
        model_forward(&ma, pool, toks[i], i);
        memcpy(seq_logits[i], ma.logits, 248320 * sizeof(float));
    }

    /* batch */
    double t0 = now_sec();
    model_forward_batch(&mb, pool, toks, 0, B);
    double dt_batch = now_sec() - t0;

    /* comparaison */
    double max_abs = 0, max_rel = 0;
    uint32_t worst_b = 0;
    int n_diff_top = 0;
    for (uint32_t b = 0; b < B; b++) {
        const float *lb = mb.logits_b + (size_t)b * 248320;
        double tmax = 0;
        for (uint32_t v = 0; v < 248320; v += 7) {   /* echantillonnage 1/7 */
            double ref = seq_logits[b][v];
            double d = fabs((double)lb[v] - ref);
            double rel = d / (fabs(ref) > 1.0 ? fabs(ref) : 1.0);
            if (d > tmax) tmax = d;
            if (d > max_abs) { max_abs = d; worst_b = b; }
            if (rel > max_rel) max_rel = rel;
        }
        uint32_t gseq = model_sample_greedy(seq_logits[b], 248320);
        uint32_t gbat = model_sample_greedy(lb, 248320);
        if (gseq != gbat) n_diff_top++;
        printf("token %u : argmax seq=%u batch=%u %s  max_abs=%.3e\n", b, gseq, gbat,
               gseq == gbat ? "OK" : "DIFF", tmax);
    }
    printf("\nmax_abs=%.3e (token %u)  max_rel=%.3e  argmax_diff=%d/4\n",
           max_abs, worst_b, max_rel, n_diff_top);
    printf("forward_batch(%u tokens) : %.0f ms (%.2f ms/token)\n",
           B, dt_batch * 1e3, dt_batch * 1e3 / B);

    pool_destroy(pool);
    model_free(&ma); model_free(&mb);
    return n_diff_top ? 1 : 0;
}
