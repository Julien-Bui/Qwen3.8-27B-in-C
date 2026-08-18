/* Harnais de test : déquantifie des lignes réelles du fichier et vérifie
 * la plausibilité statistique (poids LLM ~ N(0, petite variance)). */
#include "gguf.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

static int check_row(const gguf_context_t *ctx, const char *name, uint64_t row) {
    const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name);
    if (!t) {
        printf("%-36s ABSENT\n", name);
        return 0;
    }
    if (t->n_dims != 2 && row != 0) {
        printf("%-36s 1-D, ligne 0 seulement\n", name);
        return 0;
    }
    const uint64_t ne0 = t->ne[0];
    const char *fmt = quant_type_name(t->type);
    if (fmt[0] == 'U') {
        printf("%-36s type %u non supporté\n", name, (unsigned)t->type);
        return 0;
    }

    float *v = malloc((size_t)ne0 * sizeof *v);
    if (!v) return 0;
    if (dequant_row(t->type, quant_row_ptr(t, row), ne0, v) != 0) {
        printf("%-36s ne0=%" PRIu64 " non multiple des blocs ?\n", name, ne0);
        free(v);
        return 0;
    }

    double sum = 0, sum2 = 0;
    float mn = v[0], mx = v[0];
    int bad = 0;
    for (uint64_t i = 0; i < ne0; i++) {
        if (!isfinite(v[i])) bad = 1;
        sum  += v[i];
        sum2 += (double)v[i] * v[i];
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }
    double mean = sum / ne0;
    double std  = sqrt(sum2 / ne0 - mean * mean);

    printf("%-36s %-7s ligne %-5" PRIu64 " n=%-6" PRIu64
           " min=%+9.5f max=%+9.5f moy=%+8.5f ect=%8.5f %s\n",
           name, fmt, row, ne0, mn, mx, mean, std,
           bad ? "  << NON FINI !" : "");

    printf("    premiers: ");
    for (int i = 0; i < 8; i++) printf("%+9.6f ", v[i]);
    printf("\n");

    free(v);
    /* Poids de matrice : centrés ~0. Normes : centrées ~1-2. */
    if (strstr(name, "norm"))
        return !bad && mean > 0.0 && mean < 5.0;
    return !bad && fabs(mean) < 0.5 && fabs(mx) < 100.0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <modele.gguf>\n", argv[0]);
        return 1;
    }
    gguf_context_t ctx;
    if (gguf_open(&ctx, argv[1]) != 0) return 1;

    printf("=== Module A : déquantification de référence ===\n\n");
    int ok = 1;
    ok &= check_row(&ctx, "output_norm.weight", 0);            /* F32 */
    ok &= check_row(&ctx, "blk.0.ssm_a", 0);                   /* F32 */
    ok &= check_row(&ctx, "token_embd.weight", 0);             /* Q4_K */
    ok &= check_row(&ctx, "token_embd.weight", 12345);         /* Q4_K, loin */
    ok &= check_row(&ctx, "blk.0.attn_qkv.weight", 0);         /* Q5_K */
    ok &= check_row(&ctx, "blk.0.attn_gate.weight", 0);        /* IQ4_XS */
    ok &= check_row(&ctx, "blk.0.ffn_down.weight", 100);       /* IQ4_XS */
    ok &= check_row(&ctx, "output.weight", 0);                 /* Q6_K */
    ok &= check_row(&ctx, "blk.64.nextn.eh_proj.weight", 0);   /* Q8_0 */
    ok &= check_row(&ctx, "blk.3.attn_q.weight", 0);           /* couche attention */

    printf("\nVERDICT : %s\n", ok ? "TOUS PLAUSIBLES" : "ANOMALIE DETECTEE");
    gguf_close(&ctx);
    return ok ? 0 : 1;
}
