/* Module B : recensement des couches, validation des dimensions,
 * rapport d'allocation mémoire. */
#include "model.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <modele.gguf> [n_ctx]\n", argv[0]);
        return 1;
    }
    uint32_t n_ctx = argc >= 3 ? (uint32_t)atoi(argv[2]) : 4096;

    qwen_model_t m;
    if (model_init(&m, argv[1], n_ctx) != 0) {
        fprintf(stderr, "model_init échoué\n");
        return 1;
    }
    const qwen_dims_t *d = &m.dims;

    printf("=== Module B : gestionnaire d'état ===\n\n");
    printf("tronc=%u couches  (GDN=%u, attention=%u)  | MTP ignorée=%u\n",
           d->n_layer, d->n_recr_layer, d->n_attn_layer, m.cfg.nextn_layers);
    printf("attention : n_head=%u n_kv=%u head_dim=%u  q_proj=%u (q||gate)\n",
           d->n_head, d->n_kv_head, d->head_dim, 2 * d->n_head * d->head_dim);
    printf("GDN       : v_heads=%u k_heads=%u head_k=%u head_v=%u d_inner=%u conv_dim=%u\n",
           d->n_v_heads, d->n_k_heads, d->head_k_dim, d->head_v_dim, d->d_inner, d->conv_dim);
    printf("rope      : theta=%g dims=%u sections={%u,%u,%u,%u}\n",
           m.cfg.rope_theta, m.cfg.rope_dim_count,
           d->rope_sections[0], d->rope_sections[1], d->rope_sections[2], d->rope_sections[3]);

    /* carte des couches */
    printf("\ncouches : ");
    for (uint32_t il = 0; il < d->n_layer; il++)
        printf("%c", m.layers[il].is_recurrent ? '.' : 'A');
    printf("\n         (A = attention, . = GDN)\n");

    /* mémoire */
    const double mb = 1024.0 * 1024.0;
    size_t kv = 0, conv = 0, ssm = 0;
    for (uint32_t il = 0; il < d->n_layer; il++) {
        qwen_layer_t *L = &m.layers[il];
        if (L->is_recurrent) {
            conv += (size_t)d->conv_dim * d->d_conv * 4;
            ssm  += (size_t)d->n_v_heads * d->head_v_dim * d->head_k_dim * 4;
        } else {
            kv += 2 * (size_t)m.n_ctx * d->n_kv_head * d->head_dim * 4;
        }
    }
    printf("\nmémoire d'état (n_ctx=%u) :\n", m.n_ctx);
    printf("  KV-cache  (16 couches) : %8.1f Mo\n", kv / mb);
    printf("  conv      (48 couches) : %8.1f Mo\n", conv / mb);
    printf("  ssm       (48 couches) : %8.1f Mo\n", ssm / mb);
    printf("  activations + logits   : %8.1f Mo\n",
           ((size_t)m.cfg.vocab_size + 4 * d->n_embd + 2 * d->n_head * d->head_dim
             + d->d_inner + d->n_ff + d->n_head * d->head_dim) * 4.0 / mb);
    printf("  poids (mmap, non résident) : %.2f Go\n", m.gguf.file_size / (1024.0 * mb));

    model_free(&m);
    printf("\nOK\n");
    return 0;
}
