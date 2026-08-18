#define _POSIX_C_SOURCE 200809L

#include "model.h"
#include "tokenizer.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "Qwen3.8-27B-IQ4_XS.gguf";
    printf("=== Test Model Forward Pass on %s ===\n", model_path);

    qwen_model_t model;
    if (model_init(&model, model_path, 128) != 0) {
        fprintf(stderr, "Failed to init model\n");
        return 1;
    }

    pool_t *pool = pool_create(8, 0);

    uint32_t test_token = 151644;
    printf("Running model_forward(token=%u, pos=0) ...\n", test_token);

    double t0 = get_time_sec();
    model_forward(&model, pool, test_token, 0);
    double elapsed = get_time_sec() - t0;

    printf("Forward pass completed in %.3f ms\n", elapsed * 1000.0);

    float max_logit = -1e9f, min_logit = 1e9f;
    uint32_t top_id = model_sample_greedy(model.logits, model.cfg.vocab_size);

    for (uint32_t i = 0; i < model.cfg.vocab_size; i++) {
        if (model.logits[i] > max_logit) max_logit = model.logits[i];
        if (model.logits[i] < min_logit) min_logit = model.logits[i];
    }

    printf("Logits stats: min = %.4f, max = %.4f, greedy token = %u (logit = %.4f)\n",
           min_logit, max_logit, top_id, model.logits[top_id]);

    pool_destroy(pool);
    model_free(&model);
    printf("=== Test Forward Pass completed successfully ===\n");
    return 0;
}
