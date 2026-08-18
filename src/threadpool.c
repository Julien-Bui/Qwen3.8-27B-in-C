#define _GNU_SOURCE

#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>

struct pool;

typedef struct {
    struct pool *p;
    int          tid;
} worker_ctx_t;

struct pool {
    worker_ctx_t     *wctx;        /* n_workers entrées */
    pthread_t        *threads;     /* n_threads - 1 workers */
    int               n_total;     /* workers + thread appelant */
    pthread_barrier_t bar_start, bar_end;
    pool_fn_t         fn;
    void             *arg;
    volatile int      quit;
};

static void *worker_main(void *ud) {
    worker_ctx_t *w = ud;
    pool_t *p = w->p;

    for (;;) {
        pthread_barrier_wait(&p->bar_start);
        if (p->quit) break;
        p->fn(p->arg, w->tid, p->n_total);
        pthread_barrier_wait(&p->bar_end);
    }
    return NULL;
}

pool_t *pool_create(int n_threads, int pin) {
    if (n_threads < 1) n_threads = 1;

    pool_t *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n_total = n_threads;

    if (pthread_barrier_init(&p->bar_start, NULL, n_threads) != 0) goto fail;
    if (pthread_barrier_init(&p->bar_end,   NULL, n_threads) != 0) goto fail;

    const int n_workers = n_threads - 1;
    if (n_workers > 0) {
        p->wctx    = calloc(n_workers, sizeof *p->wctx);
        p->threads = calloc(n_workers, sizeof *p->threads);
        if (!p->wctx || !p->threads) goto fail;
        for (int k = 0; k < n_workers; k++) {
            p->wctx[k].p = p;
            p->wctx[k].tid = k + 1;
            if (pthread_create(&p->threads[k], NULL, worker_main, &p->wctx[k]) != 0)
                goto fail;
            if (pin) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(k, &set);   /* worker k -> CPU k (P-cores d'abord) */
                pthread_setaffinity_np(p->threads[k], sizeof set, &set);
            }
        }
    }
    return p;

fail:
    perror("pool_create");
    pool_destroy(p); /* sûr même partiellement initialisé */
    return NULL;
}

void pool_run(pool_t *p, pool_fn_t fn, void *arg) {
    if (p->n_total == 1) {
        fn(arg, 0, 1);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    pthread_barrier_wait(&p->bar_start);
    fn(arg, 0, p->n_total);            /* le thread appelant = tid 0 */
    pthread_barrier_wait(&p->bar_end);
}

void pool_destroy(pool_t *p) {
    if (!p) return;
    if (p->n_total > 1 && p->threads) {
        p->quit = 1;
        pthread_barrier_wait(&p->bar_start);
        for (int k = 0; k < p->n_total - 1; k++)
            if (p->threads[k]) pthread_join(p->threads[k], NULL);
    }
    pthread_barrier_destroy(&p->bar_start);
    pthread_barrier_destroy(&p->bar_end);
    free(p->threads);
    free(p->wctx);
    free(p);
}
