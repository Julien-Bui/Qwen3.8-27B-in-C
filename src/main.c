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
    printf("  -p, --prompt PROMPT    Text prompt (default: 'Hello! Introduce yourself:')\n");
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
    const char *prompt = "Hello! Introduce yourself:";
    uint32_t n_predict = 32;
    uint32_t n_ctx = 512;
    uint32_t n_threads = 12;
    float temp = 0.7f;
    float top_p = 0.9f;
    uint32_t top_k = 40;
    float repeat_penalty = 1.1f;
    int profile = 0;
    uint32_t n_spec = 0;   /* drafts MTP (0 = off, sinon 1..SPEC_MAX_B-1) */

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
        } else if (strcmp(argv[i], "--profile") == 0) {
            profile = 1;
        } else if (strcmp(argv[i], "--spec") == 0) {
            if (++i < argc) n_spec = (uint32_t)atoi(argv[i]);
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

    if (n_ctx < 2) n_ctx = 2;

    uint32_t prompt_tokens[1024];
    uint32_t n_prompt = tokenizer_encode(&tok, prompt, prompt_tokens, 1024, false);
    if (n_prompt > n_ctx) {
        fprintf(stderr, "\nWarning: prompt (%u tokens) exceeds context size (%u); truncating\n",
                n_prompt, n_ctx);
        n_prompt = n_ctx;
    }
    if ((uint64_t)n_prompt + n_predict > 2048) {
        fprintf(stderr, "\nWarning: n_predict clamped to %u (history buffer limit)\n",
                2048 - n_prompt);
        n_predict = 2048 - n_prompt;
    }
    printf("\nPrompt (%u tokens): %s\n", n_prompt, prompt);
    printf("Generation: ");
    fflush(stdout);

    uint32_t all_tokens[2048];
    size_t n_all = 0;
    for (uint32_t i = 0; i < n_prompt; i++) all_tokens[n_all++] = prompt_tokens[i];

    double t_prefill_start = get_time_sec();
    for (uint32_t pos = 0; pos < n_prompt; pos++) {
        model_forward(&model, pool, prompt_tokens[pos], pos);
        if (n_spec > 0) {
            /* MTP anchor: fills MTP KV cache and produces h_nextn */
            memcpy(model.mtp_h, model.x, model.dims.n_embd * sizeof(float));
            mtp_forward(&model, pool, prompt_tokens[pos], pos, 0);
        }
    }
    double t_prefill = get_time_sec() - t_prefill_start;

    uint32_t generated = 0;
    double t_gen_start = get_time_sec();

    if (n_spec > 0 && temp <= 0.0f) {
        /* ---- Greedy speculative decoding (MTP drafter, backbone verifier).
         * Lossless: output stream exactly matches backbone greedy decode. ---- */
        if (n_spec > SPEC_MAX_B - 1) n_spec = SPEC_MAX_B - 1;
        uint64_t n_cycles = 0, n_accepted_drafts = 0, n_acc_depth1 = 0, n_acc_depth2 = 0;
        uint64_t n_drafts_total = 0;
        const float *last_logits = model.logits;

        while (generated < n_predict) {
            const uint32_t P = model.pos - 1;   /* Position of last processed token */
            /* room = number of KV positions still writable (P+1 .. n_ctx-1).
             * Clamp the draft count so that the batched verification and the
             * MTP draft forwards never write KV entries beyond n_ctx-1. */
            const uint32_t room = n_ctx - 1 - P;
            if (room < 1) break;
            const uint32_t k = (n_spec < room - 1) ? n_spec : room - 1;
            uint32_t cand[SPEC_MAX_B];
            cand[0] = model_sample_greedy(last_logits, model.cfg.vocab_size);
            if (getenv("QWEN_SPEC_DEBUG"))
                fprintf(stderr, "[cycle %llu] P=%u cand0=%u \"%s\"\n",
                        (unsigned long long)n_cycles, P, cand[0], tokenizer_decode(&tok, cand[0]));
            if (cand[0] == tok.eos_id) break;

            /* MTP drafts: cand[1..k] */
            for (uint32_t s = 1; s <= k; s++) {
                mtp_forward(&model, pool, cand[s - 1], P + s, 1);
                cand[s] = model_sample_greedy(model.logits, model.cfg.vocab_size);
            }
            const uint32_t B = k + 1;

            /* Batched verification: single pass over model weights */
            model_forward_batch(&model, pool, cand, P + 1, B);

            /* j = first mismatch index (all accepted if j == k) */
            uint32_t j = k;
            for (uint32_t i = 0; i < k; i++) {
                const uint32_t v = model_sample_greedy(
                    model.logits_b + (size_t)i * model.cfg.vocab_size, model.cfg.vocab_size);
                if (v != cand[i + 1]) { j = i; break; }
            }

            /* Emission: cand[0..j] only. Token argmax(logits_b[j]) becomes cand[0]
             * for next cycle. Strict mathematical equivalence with standard greedy. */
            for (uint32_t i = 0; i <= j && generated < n_predict; i++) {
                printf("%s", tokenizer_decode(&tok, cand[i]));
                all_tokens[n_all++] = cand[i];
                generated++;
            }
            n_accepted_drafts += j;
            n_drafts_total += k;
            if (j >= 1) n_acc_depth1++;
            if (j >= 2) n_acc_depth2++;
            fflush(stdout);
            if (model.pos >= n_ctx) break;

            /* Rollback recurrent states if drafts rejected */
            if (j < k) {
                model_rollback_to(&model, j);
                model.pos = P + j + 2;
            }

            /* MTP anchor at new frontier */
            memcpy(model.mtp_h, model.h_save + (size_t)j * model.dims.n_embd,
                   model.dims.n_embd * sizeof(float));
            mtp_forward(&model, pool, cand[j], P + j + 1, 0);
            last_logits = model.logits_b + (size_t)j * model.cfg.vocab_size;
            n_cycles++;
        }
        if (n_cycles)
            fprintf(stderr, "[spec] %llu cycles, %llu/%llu drafts accepted (%.0f%%) [depth: %llu x1, %llu x2]\n",
                    (unsigned long long)n_cycles, (unsigned long long)n_accepted_drafts,
                    (unsigned long long)n_drafts_total,
                    100.0 * (double)n_accepted_drafts / (double)n_drafts_total,
                    (unsigned long long)n_acc_depth1, (unsigned long long)n_acc_depth2);
    } else {
        uint32_t cur_token = sampler_sample(smp, model.logits, model.cfg.vocab_size, all_tokens, n_all);
        all_tokens[n_all++] = cur_token;
        printf("%s", tokenizer_decode(&tok, cur_token));
        fflush(stdout);
        generated = 1;

        for (uint32_t i = 1; i < n_predict; i++) {
            if (cur_token == tok.eos_id || model.pos >= n_ctx) break;
            model_forward(&model, pool, cur_token, model.pos);
            cur_token = sampler_sample(smp, model.logits, model.cfg.vocab_size, all_tokens, n_all);
            all_tokens[n_all++] = cur_token;
            printf("%s", tokenizer_decode(&tok, cur_token));
            fflush(stdout);
            generated++;
        }
    }
    double t_gen = get_time_sec() - t_gen_start;

    printf("\n================================================\n");
    printf("Prefill           : %u tokens in %.2f s (%.2f tok/s)\n", n_prompt, t_prefill, (double)n_prompt / t_prefill);
    printf("Generation         : %u tokens in %.2f s (%.2f tok/s)\n", generated, t_gen, (double)generated / t_gen);
    if (profile && qwen_prof_calls > 0) {
        double total = qwen_prof_embed + qwen_prof_gdn + qwen_prof_attn + qwen_prof_head;
        printf("------------------------------------------------\n");
        printf("Profile (%llu fwd): embed %.0f ms | GDN %.0f ms (%.1f%%) | attn %.0f ms (%.1f%%) | head %.0f ms (%.1f%%)\n",
               (unsigned long long)qwen_prof_calls,
               qwen_prof_embed * 1e3,
               qwen_prof_gdn * 1e3, 100.0 * qwen_prof_gdn / total,
               qwen_prof_attn * 1e3, 100.0 * qwen_prof_attn / total,
               qwen_prof_head * 1e3, 100.0 * qwen_prof_head / total);
    }
    printf("================================================\n");

    sampler_free(smp);
    pool_destroy(pool);
    tokenizer_free(&tok);
    model_free(&model);
    return 0;
}
