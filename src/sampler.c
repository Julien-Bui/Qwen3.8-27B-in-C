#define _POSIX_C_SOURCE 200809L

#include "sampler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

typedef struct {
    float prob;
    uint32_t id;
} prob_index_t;

static int cmp_desc(const void *a, const void *b) {
    const prob_index_t *pa = (const prob_index_t *)a;
    const prob_index_t *pb = (const prob_index_t *)b;
    if (pa->prob < pb->prob) return 1;
    if (pa->prob > pb->prob) return -1;
    return 0;
}

static inline void heap_sift_down(prob_index_t *heap, uint32_t k, uint32_t i) {
    while (2 * i + 1 < k) {
        uint32_t left = 2 * i + 1;
        uint32_t right = left + 1;
        uint32_t smallest = i;
        if (heap[left].prob < heap[smallest].prob) smallest = left;
        if (right < k && heap[right].prob < heap[smallest].prob) smallest = right;
        if (smallest == i) break;
        prob_index_t tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

sampler_t *sampler_init(float temp, float top_p, uint32_t top_k, float repeat_penalty, uint64_t seed) {
    sampler_t *s = calloc(1, sizeof(sampler_t));
    if (!s) return NULL;
    s->temp = temp;
    s->top_p = top_p;
    s->top_k = (top_k == 0) ? 40 : top_k;
    s->repeat_penalty = (repeat_penalty <= 0.0f) ? 1.0f : repeat_penalty;
    s->repeat_last_n = 64;
    s->rng_state = (seed == 0) ? 0x123456789abcdefULL : seed;
    return s;
}

void sampler_free(sampler_t *s) {
    free(s);
}

uint32_t sampler_sample(sampler_t *s, float *logits, uint32_t vocab_size, const uint32_t *tokens, size_t n_tokens) {
    if (!s || !logits || vocab_size == 0) return 0;

    if (s->repeat_penalty != 1.0f && tokens && n_tokens > 0) {
        size_t start = (n_tokens > s->repeat_last_n) ? n_tokens - s->repeat_last_n : 0;
        for (size_t i = start; i < n_tokens; i++) {
            uint32_t tid = tokens[i];
            if (tid < vocab_size) {
                if (logits[tid] > 0.0f)
                    logits[tid] /= s->repeat_penalty;
                else
                    logits[tid] *= s->repeat_penalty;
            }
        }
    }

    if (s->temp <= 0.0f || s->top_k == 1) {
        uint32_t max_i = 0;
        float max_v = logits[0];
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_v) {
                max_v = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    uint32_t K = s->top_k;
    if (K > 256) K = 256;
    if (K > vocab_size) K = vocab_size;

    prob_index_t heap[256];
    uint32_t heap_size = 0;

    for (uint32_t i = 0; i < vocab_size; i++) {
        float val = logits[i];
        if (heap_size < K) {
            uint32_t cur = heap_size++;
            heap[cur].prob = val;
            heap[cur].id = i;
            while (cur > 0) {
                uint32_t parent = (cur - 1) / 2;
                if (heap[cur].prob < heap[parent].prob) {
                    prob_index_t tmp = heap[cur];
                    heap[cur] = heap[parent];
                    heap[parent] = tmp;
                    cur = parent;
                } else break;
            }
        } else if (val > heap[0].prob) {
            heap[0].prob = val;
            heap[0].id = i;
            heap_sift_down(heap, K, 0);
        }
    }

    qsort(heap, K, sizeof(prob_index_t), cmp_desc);

    float max_l = heap[0].prob / s->temp;
    double sum = 0.0;
    for (uint32_t i = 0; i < K; i++) {
        heap[i].prob = expf((heap[i].prob / s->temp) - max_l);
        sum += heap[i].prob;
    }
    float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (uint32_t i = 0; i < K; i++) {
        heap[i].prob *= inv_sum;
    }

    uint32_t n_cand = K;
    if (s->top_p < 1.0f) {
        float cumulative = 0.0f;
        uint32_t last_idx = 0;
        for (uint32_t i = 0; i < K; i++) {
            cumulative += heap[i].prob;
            last_idx = i;
            if (cumulative >= s->top_p) break;
        }
        n_cand = last_idx + 1;
    }

    float total_p = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        total_p += heap[i].prob;
    }

    uint64_t rand_u64 = xorshift64(&s->rng_state);
    float r = ((float)(rand_u64 % 10000000) / 10000000.0f) * total_p;

    float acc = 0.0f;
    uint32_t selected = heap[0].id;
    for (uint32_t i = 0; i < n_cand; i++) {
        acc += heap[i].prob;
        if (acc >= r) {
            selected = heap[i].id;
            break;
        }
    }

    return selected;
}
