#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    float temp;
    float top_p;
    uint32_t top_k;
    float repeat_penalty;
    uint32_t repeat_last_n;
    uint64_t rng_state;
} sampler_t;

sampler_t *sampler_init(float temp, float top_p, uint32_t top_k, float repeat_penalty, uint64_t seed);
void sampler_free(sampler_t *s);

uint32_t sampler_sample(sampler_t *s, float *logits, uint32_t vocab_size, const uint32_t *tokens, size_t n_tokens);

#endif /* SAMPLER_H */
