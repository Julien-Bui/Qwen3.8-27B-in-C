#define _GNU_SOURCE

#include "model.h"
#include "kernels.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ============ PARTIE 1 : exactitude AVX2 vs référence scalaire ============ */
static int check_format(const qwen_model_t *m, const char *name, uint64_t rows_to_test) {
    const gguf_tensor_info_t *t = gguf_find_tensor(&m->gguf, name);
    if (!t) { printf("%-34s ABSENT\n", name); return 1; }
    const uint64_t ne0 = t->ne[0];
    if (ne0 % QK_K != 0 && t->type != GGML_F32) {
        printf("%-34s ne0 non multiple de 256, sauté\n", name);
        return 1;
    }

    float *ref  = malloc((size_t)ne0 * sizeof *ref);
    float *out  = malloc((size_t)rows_to_test * sizeof *out);
    if (!ref || !out) return 1;

    const float *x = m->xb;
    double max_abs = 0, max_rel = 0;
    for (uint64_t r = 0; r < rows_to_test && r < t->ne[1]; r++) {
        if (dequant_row(t->type, quant_row_ptr(t, r), ne0, ref) != 0) break;
        double s = 0;
        for (uint64_t i = 0; i < ne0; i++) s += (double)ref[i] * x[i];
        gemv_rows(t, x, out, r, r + 1);
        double diff = fabs(s - out[r]);
        if (diff > max_abs) max_abs = diff;
        double rel = diff / (fabs(s) + 1e-9);
        if (rel > max_rel) max_rel = rel;
    }
    printf("%-34s %-7s max_abs=%.3e  max_rel=%.3e  %s\n",
           name, quant_type_name(t->type), max_abs, max_rel,
           max_rel < 1e-4 ? "OK" : "ECHEC");
    free(ref); free(out);
    return max_rel < 1e-4 ? 0 : 1;
}

/* ============ PARTIE 2 : balayage GEMV = un pas de décodage ============ */
static double sweep(const qwen_model_t *m, pool_t *pool) {
    const double t0 = now_s();
    float *x = m->xb;
    float *trash = m->logits; /* 248320 floats : assez pour toute sortie */

    for (uint32_t il = 0; il < m->dims.n_layer; il++) {
        const qwen_layer_t *L = &m->layers[il];
        if (L->is_recurrent) {
            gemv(pool, L->wqkv, x, trash);
            gemv(pool, L->wqkv_gate, x, trash);
            gemv(pool, L->ssm_beta, x, trash);
            gemv(pool, L->ssm_alpha, x, trash);
            gemv(pool, L->ssm_out, x, trash);
        } else {
            gemv(pool, L->attn_q, x, trash);
            gemv(pool, L->attn_k, x, trash);
            gemv(pool, L->attn_v, x, trash);
            gemv(pool, L->attn_out, x, trash);
        }
        gemv(pool, L->ffn_gate, x, trash);
        gemv(pool, L->ffn_up, x, trash);
        gemv(pool, L->ffn_down, x, trash);
    }
    gemv(pool, m->output, x, trash);   /* tête LM : 248320 lignes */
    return now_s() - t0;
}

static double sweep_bytes(const qwen_model_t *m) {
    double bytes = 0;
    for (uint32_t il = 0; il < m->dims.n_layer; il++) {
        const qwen_layer_t *L = &m->layers[il];
        const gguf_tensor_info_t *mats[8];
        int n = 0;
        if (L->is_recurrent) {
            mats[n++] = L->wqkv; mats[n++] = L->wqkv_gate;
            mats[n++] = L->ssm_beta; mats[n++] = L->ssm_alpha;
            mats[n++] = L->ssm_out;
        } else {
            mats[n++] = L->attn_q; mats[n++] = L->attn_k;
            mats[n++] = L->attn_v; mats[n++] = L->attn_out;
        }
        mats[n++] = L->ffn_gate; mats[n++] = L->ffn_up; mats[n++] = L->ffn_down;
        for (int k = 0; k < n; k++)
            bytes += (double)quant_row_bytes(mats[k]->type, mats[k]->ne[0]) * mats[k]->ne[1];
    }
    bytes += (double)quant_row_bytes(m->output->type, m->output->ne[0]) * m->output->ne[1];
    return bytes;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <modele.gguf> [threads_testes...]\n", argv[0]);
        return 1;
    }

    qwen_model_t m;
    if (model_init(&m, argv[1], 64) != 0) return 1;

    /* x pseudo-aléatoire déterministe */
    unsigned seed = 42;
    for (uint32_t i = 0; i < m.dims.n_embd; i++) {
        seed = seed * 1664525u + 1013904223u;
        m.xb[i] = ((int)(seed >> 28) - 8) * 0.01f;
    }

    printf("=== Module C : kernels AVX2 ===\n\n[1] exactitude (AVX2 vs référence scalaire)\n");
    int fails = 0;
    fails += check_format(&m, "blk.3.attn_q.weight", 8);            /* IQ4_XS */
    fails += check_format(&m, "blk.0.ffn_down.weight", 8);          /* IQ4_XS */
    fails += check_format(&m, "blk.0.attn_qkv.weight", 8);          /* Q5_K   */
    fails += check_format(&m, "token_embd.weight", 4);              /* Q4_K   */
    fails += check_format(&m, "output.weight", 4);                  /* Q6_K   */
    fails += check_format(&m, "blk.64.nextn.eh_proj.weight", 4);    /* Q8_0   */
    fails += check_format(&m, "blk.0.ssm_conv1d.weight", 4);        /* F32    */
    if (fails) { printf("\n%d ECHEC(S) — kernels invalides\n", fails); return 1; }

    /* [2] benchmark : les threads à tester */
    int nts[16], n_test = 0;
    if (argc >= 3) {
        for (int i = 2; i < argc && n_test < 16; i++) nts[n_test++] = atoi(argv[i]);
    } else {
        int def[] = { 1, 6, 8, 12, 20 };
        for (unsigned i = 0; i < sizeof def / sizeof def[0]; i++) nts[n_test++] = def[i];
    }

    const double bytes = sweep_bytes(&m);
    printf("\n[2] balayage GEMV complet (%.2f Go de poids par token)\n", bytes / 1e9);
    printf("%9s  %12s  %10s  %10s\n", "threads", "temps (ms)", "Go/s", "tok/s est.");

    double best = 1e9;
    for (int k = 0; k < n_test; k++) {
        const int n = nts[k];
        pool_t *pool = pool_create(n, 1);
        if (!pool) { printf("%d threads : échec pool\n", n); continue; }
        /* épingler aussi le thread principal (le dernier CPU) */
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(n - 1, &set);
        sched_setaffinity(0, sizeof set, &set);

        if (k == 0) {
            printf("          (échauffement : lecture des %.2f Go depuis le NVMe/page cache...)\n",
                   bytes / 1e9);
            fflush(stdout);
            sweep(&m, pool); /* première passe : pages résidentes */
        }
        double t = 1e9;
        for (int rep = 0; rep < 2; rep++) {
            double d = sweep(&m, pool);
            if (d < t) t = d;
        }
        if (t < best) best = t;
        printf("%9d  %12.1f  %10.2f  %10.2f\n", n, t * 1e3, bytes / t / 1e9, 1.0 / t);
        fflush(stdout);
        pool_destroy(pool);
        CPU_ZERO(&set);
        for (int c = 0; c < 20; c++) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof set, &set);
    }

    printf("\nmeilleur : %.2f Go/s -> ~%.1f tok/s en décodage (hors attention/SSM)\n",
           bytes / best / 1e9, 1.0 / best);
    model_free(&m);
    return 0;
}
