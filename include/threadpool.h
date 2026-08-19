#ifndef THREADPOOL_H
#define THREADPOOL_H

/* Persistent threadpool: n_workers threads + calling thread participates (tid 0). */
typedef struct pool pool_t;

/* n_threads = total thread count (worker 0 = caller thread).
 * pin != 0 : pins each worker k to CPU k-1. */
pool_t *pool_create(int n_threads, int pin);
void    pool_destroy(pool_t *p);

typedef void (*pool_fn_t)(void *arg, int tid, int ntotal);

/* Blocks until work has completed across all threads. */
void pool_run(pool_t *p, pool_fn_t fn, void *arg);

#endif /* THREADPOOL_H */
