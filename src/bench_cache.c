/* Micro-bench : débit des kernels sur données résidentes en cache L3. */
#define _POSIX_C_SOURCE 200809L

#include "model.h"
#include "kernels.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(const qwen_model_t *m, const char *name, uint64_t rows, int reps) {
    const gguf_tensor_info_t *t = gguf_find_tensor(&m->gguf, name);
    if (!t) return;
    float *out = malloc(rows * sizeof *out);
    double bytes = (double)quant_row_bytes(t->type, t->ne[0]) * rows;
    double t0 = now_s();
    for (int i = 0; i < reps; i++)
        gemv_rows(t, m->xb, out, 0, rows);
    double dt = (now_s() - t0) / reps;
    printf("%-30s %6.1f Mo  %8.2f Go/s  (%.1f Mo cache L3 attendu)\n",
           name, bytes / 1e6, bytes / dt / 1e9, bytes / 1e6);
    free(out);
}

int main(int argc, char **argv) {
    qwen_model_t m;
    if (model_init(&m, argv[1], 64) != 0) return 1;
    unsigned seed = 7;
    for (uint32_t i = 0; i < m.dims.n_embd; i++) {
        seed = seed * 1664525u + 1013904223u;
        m.xb[i] = ((int)(seed >> 28) - 8) * 0.01f;
    }
    printf("=== kernels sur données résidentes (mono-thread) ===\n");
    bench(&m, "blk.0.attn_qkv.weight", 512, 200);    /* Q5_K, ~3.5 Mo */
    bench(&m, "blk.0.ffn_gate.weight", 512, 200);    /* IQ4_XS, ~4.3 Mo */
    bench(&m, "output.weight", 512, 200);            /* Q6_K, ~5.3 Mo */
    printf("L3 du 12900H : 24 Mo — ces tranches devraient y tenir.\n");
    model_free(&m);
    return 0;
}
