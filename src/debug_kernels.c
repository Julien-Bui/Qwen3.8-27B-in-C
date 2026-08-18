/* Debug 3 : sonde élément par élément du kernel AVX2 via vecteurs unitaires. */
#include "model.h"
#include "kernels.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    qwen_model_t m;
    if (model_init(&m, argv[1], 64) != 0) return 1;

    const gguf_tensor_info_t *t = gguf_find_tensor(&m.gguf, "token_embd.weight");
    const uint64_t ne0 = t->ne[0];
    float *y = malloc(ne0 * sizeof *y);
    dequant_row(t->type, quant_row_ptr(t, 0), ne0, y);

    float *x = calloc(ne0, sizeof *x);
    float out;
    int first_bad[8]; int n_bad = 0;
    for (uint64_t i = 0; i < ne0 && n_bad < 8; i++) {
        x[i] = 1.0f;
        out = 0;
        gemv_rows(t, x, &out, 0, 1);
        x[i] = 0.0f;
        if (fabs(out - y[i]) > 1e-4) first_bad[n_bad++] = (int)i;
    }
    printf("token_embd row0 : %d premiers éléments divergents (sur 5120 sondés) :\n", n_bad);
    for (int k = 0; k < n_bad; k++) {
        int i = first_bad[k];
        printf("  elem %4d (bloc %d, pos %3d) : scalaire=%+9.5f  avx2=%+9.5f\n",
               i, i / 256, i % 256, y[i], (float){0});
    }
    /* version avec valeur affichée correcte */
    for (int k = 0; k < n_bad; k++) {
        int i = first_bad[k];
        x[i] = 1.0f; out = 0;
        gemv_rows(t, x, &out, 0, 1);
        x[i] = 0.0f;
        printf("  elem %4d : scalaire=%+9.5f  avx2=%+9.5f  diff=%+9.5f\n",
               i, y[i], out, out - y[i]);
    }
    model_free(&m);
    return 0;
}
