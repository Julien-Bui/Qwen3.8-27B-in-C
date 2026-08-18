#ifndef THREADPOOL_H
#define THREADPOOL_H

/* Pool persistant : n_workers threads + le thread appelant participe
 * (tid 0). Le job reçoit (tid, ntotal) et traite une tranche contiguë. */

typedef struct pool pool_t;

/* n_threads = nombre total d'exécutants (worker n°0 = thread appelant).
 * pin != 0 : chaque worker k est épinglé sur le CPU k-1 (P-cores en premier
 * sur l'énumération Linux d'Alder Lake). */
pool_t *pool_create(int n_threads, int pin);
void    pool_destroy(pool_t *p);

typedef void (*pool_fn_t)(void *arg, int tid, int ntotal);

/* Bloque jusqu'à ce que le job soit exécuté par tous les threads. */
void pool_run(pool_t *p, pool_fn_t fn, void *arg);

#endif
