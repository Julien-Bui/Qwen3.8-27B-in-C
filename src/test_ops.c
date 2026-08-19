#define _GNU_SOURCE

#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define ANSI_GREEN "\033[0;32m"
#define ANSI_RED   "\033[0;31m"
#define ANSI_BOLD  "\033[1m"
#define ANSI_RESET "\033[0m"

static int g_fails = 0;
static int g_tests = 0;

#define CHECK_REL(desc, val, ref, tol) do { \
    g_tests++; \
    double diff = fabs((double)(val) - (double)(ref)); \
    double denom = fabs((double)(ref)); \
    double rel = (denom > 1e-7) ? (diff / denom) : diff; \
    if (rel > (tol) || isnan(val) || isinf(val)) { \
        printf("  " ANSI_RED "[FAIL]" ANSI_RESET " %s: val=%.8e ref=%.8e diff=%.3e rel=%.3e (tol=%.1e)\n", \
               (desc), (double)(val), (double)(ref), diff, rel, (double)(tol)); \
        g_fails++; \
    } \
} while (0)

#define ASSERT_TRUE(desc, cond) do { \
    g_tests++; \
    if (!(cond)) { \
        printf("  " ANSI_RED "[FAIL]" ANSI_RESET " %s\n", (desc)); \
        g_fails++; \
    } \
} while (0)

/* ============== Referentiel scalaire en double precision ============== */

static void ref_rmsnorm(double *out, const float *x, const float *weight, uint64_t n, double eps) {
    double sum_sq = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double val = (double)x[i];
        sum_sq += val * val;
    }
    double mean_sq = sum_sq / (double)n;
    double scale = 1.0 / sqrt(mean_sq + eps);
    for (uint64_t i = 0; i < n; i++) {
        out[i] = (double)x[i] * scale * (weight ? (double)weight[i] : 1.0);
    }
}

static void ref_l2norm(double *out, const float *x, uint64_t n, double eps) {
    double sum_sq = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double val = (double)x[i];
        sum_sq += val * val;
    }
    double norm = sqrt(sum_sq);
    double scale = 1.0 / (norm > eps ? norm : eps);
    for (uint64_t i = 0; i < n; i++) {
        out[i] = (double)x[i] * scale;
    }
}

static double ref_silu(double x) {
    return x / (1.0 + exp(-x));
}

static double ref_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double ref_softplus(double x) {
    if (x > 30.0) return x;
    if (x < -50.0) return exp(x);
    return log1p(exp(x));
}

static void ref_softmax(double *out, const float *x, uint64_t n) {
    double max_val = -1e30;
    for (uint64_t i = 0; i < n; i++) {
        if ((double)x[i] > max_val) max_val = (double)x[i];
    }
    double sum_exp = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        out[i] = exp((double)x[i] - max_val);
        sum_exp += out[i];
    }
    double inv_sum = 1.0 / sum_exp;
    for (uint64_t i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

/* ============== Tests specifiques ============== */

static void test_rmsnorm_known(void) {
    printf("  [+] Test RMSNorm known vectors\n");
    float x1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float w1[4] = { 2.0f, 2.0f, 2.0f, 2.0f };
    float out1[4] = { 0 };
    rmsnorm(out1, x1, w1, 4, 0.0f);
    for (int i = 0; i < 4; i++) {
        CHECK_REL("RMSNorm unitaire x=1 w=2", out1[i], 2.0, 1e-6);
    }

    float x2[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
    float w2[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float out2[4] = { 0 };
    rmsnorm(out2, x2, w2, 4, 0.0f);
    CHECK_REL("RMSNorm signe +", out2[0], 1.0, 1e-6);
    CHECK_REL("RMSNorm signe -", out2[1], -1.0, 1e-6);
}

static void test_l2norm_known(void) {
    printf("  [+] Test L2Norm known vectors\n");
    float x1[2] = { 3.0f, 4.0f };
    float out1[2] = { 0 };
    l2norm(out1, x1, 2, 1e-5f);
    CHECK_REL("L2Norm [3, 4] -> 3/5", out1[0], 0.6, 1e-6);
    CHECK_REL("L2Norm [3, 4] -> 4/5", out1[1], 0.8, 1e-6);

    /* Cas vecteur zero avec eps */
    float x_zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float out_zero[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    l2norm(out_zero, x_zero, 4, 1e-5f);
    for (int i = 0; i < 4; i++) {
        CHECK_REL("L2Norm zeros", out_zero[i], 0.0, 1e-6);
    }
}

static void test_activations_known(void) {
    printf("  [+] Test activations special values & stability\n");
    /* SiLU */
    float x_silu[4] = { 0.0f, 2.0f, -20.0f, 20.0f };
    silu_inplace(x_silu, 4);
    CHECK_REL("SiLU(0)", x_silu[0], 0.0, 1e-6);
    CHECK_REL("SiLU(2)", x_silu[1], 2.0 / (1.0 + exp(-2.0)), 1e-5);
    CHECK_REL("SiLU(-20)", x_silu[2], -20.0 * exp(-20.0), 1e-4);
    CHECK_REL("SiLU(20)", x_silu[3], 20.0, 1e-4);

    /* Sigmoid */
    float x_sig[5] = { 0.0f, 100.0f, -100.0f, 1.0f, -1.0f };
    sigmoid_inplace(x_sig, 5);
    CHECK_REL("Sigmoid(0)", x_sig[0], 0.5, 1e-6);
    CHECK_REL("Sigmoid(100)", x_sig[1], 1.0, 1e-6);
    CHECK_REL("Sigmoid(-100)", x_sig[2], 0.0, 1e-6);
    CHECK_REL("Sigmoid(1)", x_sig[3], 1.0 / (1.0 + exp(-1.0)), 1e-6);
    CHECK_REL("Sigmoid(-1)", x_sig[4], 1.0 / (1.0 + exp(1.0)), 1e-6);

    /* Softplus - verification debordement exp(x) sans NaN/Inf */
    float x_sp[6] = { 0.0f, 1.0f, -50.0f, 50.0f, 100.0f, 1000.0f };
    softplus_inplace(x_sp, 6);
    CHECK_REL("Softplus(0)", x_sp[0], log(2.0), 1e-6);
    CHECK_REL("Softplus(1)", x_sp[1], log1p(exp(1.0)), 1e-6);
    CHECK_REL("Softplus(-50)", x_sp[2], exp(-50.0), 1e-5);
    CHECK_REL("Softplus(50)", x_sp[3], 50.0, 1e-5);
    CHECK_REL("Softplus(100)", x_sp[4], 100.0, 1e-5);
    CHECK_REL("Softplus(1000)", x_sp[5], 1000.0, 1e-5);
    ASSERT_TRUE("Softplus pas de NaN/Inf", !isnan(x_sp[5]) && !isinf(x_sp[5]));

    /* Softmax avec grands nombres */
    float x_sm[4] = { 1000.0f, 1000.0f, 1000.0f, 1000.0f };
    softmax_inplace(x_sm, 4);
    for (int i = 0; i < 4; i++) {
        CHECK_REL("Softmax grand max egal", x_sm[i], 0.25, 1e-6);
    }
}

static void test_random_dimensions(void) {
    printf("  [+] Comparative multi-dimension benchmark (AVX2 & Scalar vs FP64)\n");
    const uint64_t dims[] = { 1, 3, 7, 8, 15, 16, 32, 64, 128, 256, 512, 5120 };
    const size_t n_dims = sizeof dims / sizeof dims[0];

    unsigned seed = 12345;
    for (size_t d = 0; d < n_dims; d++) {
        const uint64_t n = dims[d];
        float *x = malloc(n * sizeof *x);
        float *w = malloc(n * sizeof *w);
        float *out_c = malloc(n * sizeof *out_c);
        double *out_ref = malloc(n * sizeof *out_ref);

        for (uint64_t i = 0; i < n; i++) {
            seed = seed * 1664525u + 1013904223u;
            x[i] = ((int)(seed >> 16) - 32768) / 3276.8f;
            seed = seed * 1664525u + 1013904223u;
            w[i] = 0.5f + ((seed >> 16) & 0xFFF) / 4096.0f;
        }

        /* 1. RMSNorm */
        const float eps = 1e-6f;
        rmsnorm(out_c, x, w, n, eps);
        ref_rmsnorm(out_ref, x, w, n, (double)eps);
        for (uint64_t i = 0; i < n; i++) {
            CHECK_REL("RMSNorm dim test", out_c[i], out_ref[i], 1e-4);
        }

        /* 2. L2Norm */
        l2norm(out_c, x, n, eps);
        ref_l2norm(out_ref, x, n, (double)eps);
        for (uint64_t i = 0; i < n; i++) {
            CHECK_REL("L2Norm dim test", out_c[i], out_ref[i], 1e-4);
        }

        /* 3. Softmax */
        memcpy(out_c, x, n * sizeof *x);
        softmax_inplace(out_c, n);
        ref_softmax(out_ref, x, n);
        double sum = 0.0;
        for (uint64_t i = 0; i < n; i++) {
            CHECK_REL("Softmax dim test", out_c[i], out_ref[i], 1e-4);
            sum += out_c[i];
        }
        CHECK_REL("Softmax sum == 1.0", (float)sum, 1.0, 1e-5);

        /* 4. SiLU */
        memcpy(out_c, x, n * sizeof *x);
        silu_inplace(out_c, n);
        for (uint64_t i = 0; i < n; i++) {
            CHECK_REL("SiLU dim test", out_c[i], ref_silu(x[i]), 1e-4);
        }

        /* 5. Sigmoid */
        memcpy(out_c, x, n * sizeof *x);
        sigmoid_inplace(out_c, n);
        for (uint64_t i = 0; i < n; i++) {
            CHECK_REL("Sigmoid dim test", out_c[i], ref_sigmoid(x[i]), 1e-4);
        }

        /* 6. Softplus */
        memcpy(out_c, x, n * sizeof *x);
        softplus_inplace(out_c, n);
        for (uint64_t i = 0; i < n; i++) {
            CHECK_REL("Softplus dim test", out_c[i], ref_softplus(x[i]), 1e-4);
        }

        free(x); free(w); free(out_c); free(out_ref);
    }
}

int main(void) {
    printf(ANSI_BOLD "=== Module D1: Scalar and Vectorized Ops Test Suite ===" ANSI_RESET "\n\n");
    test_rmsnorm_known();
    test_l2norm_known();
    test_activations_known();
    test_random_dimensions();

    printf("\nTotal assertions: %d, Failures: %d\n", g_tests, g_fails);
    if (g_fails == 0) {
        printf(ANSI_GREEN ANSI_BOLD "ALL D1 TESTS PASSED SUCCESSFULLY\n" ANSI_RESET);
        return 0;
    } else {
        printf(ANSI_RED ANSI_BOLD "%d ECHEC(S) DETECTE(S)\n" ANSI_RESET, g_fails);
        return 1;
    }
}
