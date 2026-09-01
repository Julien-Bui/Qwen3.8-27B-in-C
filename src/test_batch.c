/* Test gemv_batch: verify exact match with single-token GEMV */
#include "model.h"
#include "kernels.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    qwen_model_t m;
    if (model_init(&m, argv[1], 8) != 0) return 1;
    pool_t *pool = pool_create(12, 1);

    const char *names[] = {
        "blk.3.attn_q.weight",      /* IQ4_XS */
        "blk.0.attn_qkv.weight",    /* Q5_K */
        "token_embd.weight",        /* Q4_K */
        "output.weight",            /* Q6_K */
        "blk.64.nextn.eh_proj.weight", /* Q8_0 */
        "blk.0.ssm_conv1d.weight",  /* F32 */
    };
    int n_fail = 0;
    srand(42);
    const int NB = GEMV_MAX_B;

    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
        const gguf_tensor_info_t *W = gguf_find_tensor(&m.gguf, names[k]);
        if (!W) { printf("%-32s ABSENT\n", names[k]); continue; }
        const uint64_t ne0 = W->ne[0], rows = W->ne[1];

        float *x = malloc(NB * ne0 * sizeof(float));
        float *ref = malloc(NB * (size_t)rows * sizeof(float));
        float *got = malloc(NB * (size_t)rows * sizeof(float));
        float *single = malloc(rows * sizeof(float));
        for (uint64_t i = 0; i < NB * ne0; i++) x[i] = (float)(rand() % 2000 - 1000) / 331.0f;

        /* reference : n appels mono-token */
        for (int t = 0; t < NB; t++) {
            gemv(pool, W, x + t * ne0, single);
            memcpy(ref + t * rows, single, rows * sizeof(float));
        }

        for (int n = 1; n <= NB; n++) {
            memset(got, 0, NB * (size_t)rows * sizeof(float));
            gemv_batch(pool, W, x, ne0, n, got, rows);
            int bad = 0;
            double maxd = 0;
            for (int t = 0; t < n; t++)
                for (uint64_t r = 0; r < rows; r++) {
                    double d = fabs((double)got[t * rows + r] - ref[t * rows + r]);
                    if (d > 0) bad++;
                    if (d > maxd) maxd = d;
                }
            printf("%-32s type=%2u n=%d : %s (maxdiff=%.3e)\n", names[k], (unsigned)W->type, n,
                   bad ? "DIFF" : "identical", maxd);
            if (bad) n_fail++;
        }
        free(x); free(ref); free(got); free(single);
    }

    pool_destroy(pool);
    model_free(&m);
    printf(n_fail ? "\nECHEC : %d differences\n" : "\nOK: batch == single-token everywhere\n", n_fail);
    return n_fail ? 1 : 0;
}
