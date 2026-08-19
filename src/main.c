#define _POSIX_C_SOURCE 200809L

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH       Path to GGUF model file\n");
    printf("  -p, --prompt PROMPT    Text prompt (default: 'Bonjour ! Qui es-tu ?')\n");
    printf("  -n, --n-predict N    Number of tokens to generate (default: 32)\n");
    printf("  -c, --ctx-size N     Context size (default: 512)\n");
    printf("  -t, --threads N      Number of threads (default: 8)\n");
    printf("  --temp FLOAT         Temperature (default: 0.7)\n");
    printf("  --top-p FLOAT        Top-p sampling (default: 0.9)\n");
    printf("  --top-k INT          Top-k sampling (default: 40)\n");
    printf("  --repeat-penalty FLOAT Repetition penalty (default: 1.1)\n");
}

int main(int argc, char **argv) {
    const char *model_path = "Qwen3.8-27B-IQ4_XS.gguf";
    const char *prompt = "Bonjour ! Qui es-tu ?";
    uint32_t n_predict = 32;
    uint32_t n_ctx = 512;
    uint32_t n_threads = 12;
    float temp = 0.7f;
    float top_p = 0.9f;
    uint32_t top_k = 40;
    float repeat_penalty = 1.1f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) model_path = argv[i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i < argc) prompt = argv[i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) {
            if (++i < argc) n_predict = (uint32_t)atoi(argv[i]);
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx-size") == 0) {
            if (++i < argc) n_ctx = (uint32_t)atoi(argv[i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i < argc) n_threads = (uint32_t )atoi(argv[i]);
        } else if (strcmp(argv[i], "--temp") == 0) {
            if (++i < argc) temp = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (++i < argc) top_p = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (++i < argc) top_k = (uint32_t)atoi(argv[i]);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0) {
            if (++i < argc) repeat_penalty = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("================================================\n");
    printf("  Qwen3.8-27B Inference Engine (C + AVX2 + GDN)\n");
    printf("================================================\n");
    printf("Model           : %s\n", model_path);
    printf("Threads         : %u\n", n_threads);
    printf("Context Size    : %u\n", n_ctx);
    printf("Sampling        : Temp=%.2f, TopP=%.2f, TopK=%u, RepeatPenalty=%.2f\n", temp, top_p, top_k, repeat_penalty);

    double t0 = get_time_sec();
    qwen_model_t model;
    if (model_init(&model, model_path, n_ctx) != 0) {
        fprintf(stderr, "Error loading model %s\n", model_path);
        return 1;
    }
    double t_model = get_time_sec() - t0;
    printf("Model loaded in %.2f s (%u layers: %u GDN + %u Attention)\n", t_model, model.dims.n_layer, model.dims.n_recr_layer, model.dims.n_attn_layer);

    tokenizer_t tok;
    if (tokenizer_init(&tok, &model.gguf) != 0) {
        fprintf(stderr, "Error loading tokenizer\n");
        model_free(&model);
        return 1;
    }
    printf("Tokenizer loaded (%u vocab tokens, %u BPE merges)\n", tok.vocab_size, tok.n_merges);

    pool_t *pool = pool_create(n_threads, 1);
    if (!pool) {
        fprintf(stderr, "Error creating thread pool\n");
        tokenizer_free(&tok);
        model_free(&model);
        return 1;
    }

    sampler_t *smp = sampler_init(temp, top_p, top_k, repeat_penalty, (uint64_t)time(NULL));

    uint32_t prompt_tokens[1024];
    uint32_t n_prompt = tokenizer_encode(&tok, prompt, prompt_tokens, 1024, false);
    printf("\nPrompt (%u tokens): %s\n", n_prompt, prompt);
    printf("Generation: ");
    fflush(stdout);

    uint32_t all_tokens[2048];
    size_t n_all = 0;
    for (uint32_t i = 0; i < n_prompt; i++) all_tokens[n_all++] = prompt_tokens[i];

    double t_prefill_start = get_time_sec();
    for (uint32_t pos = 0; pos < n_prompt; pos++) {
        model_forward(&model, pool, prompt_tokens[pos], pos);
    }
    double t_prefill = get_time_sec() - t_prefill_start;

    uint32_t cur_token = sampler_sample(smp, model.logits, model.cfg.vocab_size, all_tokens, n_all);
    all_tokens[n_all++] = cur_token;
    printf("%s", tokenizer_decode(&tok, cur_token));
    fflush(stdout);

    uint32_t generated = 1;
    double t_gen_start = get_time_sec();

    for (uint32_t i = 1; i < n_predict; i++) {
        if (cur_token == tok.eos_id || model.pos >= n_ctx) break;
        model_forward(&model, pool, cur_token, model.pos);
        cur_token = sampler_sample(smp, model.logits, model.cfg.vocab_size, all_tokens, n_all);
        all_tokens[n_all++] = cur_token;
        printf("%s", tokenizer_decode(&tok, cur_token));
        fflush(stdout);
        generated++;
    }
    double t_gen = get_time_sec() - t_gen_start;

    printf("\n\n================================================\n");
    printf("Prefill           : %u tokens in %.2f s (%.2f tok/s)\n", n_prompt, t_prefill, (double)n_prompt / t_prefill);
    printf("Generation         : %u tokens in %.2f s (%.2f tok/s)\n", generated, t_gen, (double)generated / t_gen);
    printf("================================================\n");

    sampler_free(smp);
    pool_destroy(pool);
    tokenizer_free(&tok);
    model_free(&model);
    return 0;
}
