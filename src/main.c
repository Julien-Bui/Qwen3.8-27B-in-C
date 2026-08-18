#include "gguf.h"
#include <stdio.h>
#include <inttypes.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <modele.gguf>\n", argv[0]);
        return 1;
    }

    gguf_context_t ctx;
    if (gguf_open(&ctx, argv[1]) != 0) {
        fprintf(stderr, "échec du chargement de %s\n", argv[1]);
        return 1;
    }

    printf("GGUF v%u | %" PRIu64 " tenseurs | arch=%s | alignement=%u | %.2f Go\n\n",
           ctx.header.version, ctx.tensor_count, ctx.arch, ctx.alignment,
           (double)ctx.file_size / (1024.0 * 1024.0 * 1024.0));

    printf("n_embd=%u n_head=%u n_head_kv=%u head_dim=%u n_layer=%u n_ff=%u\n",
           ctx.config.n_embd, ctx.config.n_head, ctx.config.n_head_kv,
           ctx.config.head_dim, ctx.config.n_layer, ctx.config.n_ff);
    printf("vocab=%u ctx_len=%u rope_theta=%g rope_dims=%u eps=%g tied=%d\n",
           ctx.config.vocab_size, ctx.config.context_len,
           ctx.config.rope_theta, ctx.config.rope_dim_count,
           ctx.config.rms_norm_eps, (int)ctx.config.tie_word_embeddings);
    printf("full_attn_interval=%u | ssm: state=%u conv=%u inner=%u groups=%u | nextn=%u\n",
           ctx.config.full_attn_interval, ctx.config.ssm_state_size,
           ctx.config.ssm_conv_kernel, ctx.config.ssm_inner_size,
           ctx.config.ssm_group_count, ctx.config.nextn_layers);

    if (argc >= 3) { /* mode dump complet : --tensors */
        for (uint64_t i = 0; i < ctx.tensor_count; i++) {
            const gguf_tensor_info_t *t = &ctx.tensors[i];
            printf("%-36s type=%2u dims=%u [", t->name, (unsigned)t->type, t->n_dims);
            for (uint32_t d = 0; d < t->n_dims; d++)
                printf("%" PRIu64 "%s", t->ne[d], d + 1 < t->n_dims ? "," : "");
            printf("]\n");
        }
        gguf_close(&ctx);
        return 0;
    }

    printf("--- 8 premiers tenseurs ---\n");
    for (uint64_t i = 0; i < ctx.tensor_count && i < 8; i++) {
        const gguf_tensor_info_t *t = &ctx.tensors[i];
        printf("%-32s type=%2u dims=%u [", t->name, (unsigned)t->type, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            printf("%" PRIu64 "%s", t->ne[d], d + 1 < t->n_dims ? "," : "");
        printf("] offset=%" PRIu64 "\n", t->offset);
    }

    printf("\n--- sondes ---\n");
    const char *probe[] = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.ffn_up.weight",
    };
    for (size_t i = 0; i < sizeof probe / sizeof probe[0]; i++) {
        const gguf_tensor_info_t *t = gguf_find_tensor(&ctx, probe[i]);
        if (t)
            printf("%-24s OK  (type=%u, ne[0]=%" PRIu64 ", ne[1]=%" PRIu64 ")\n",
                   probe[i], (unsigned)t->type, t->ne[0], t->ne[1]);
        else
            printf("%-24s ABSENT\n", probe[i]);
    }

    gguf_close(&ctx);
    return 0;
}
