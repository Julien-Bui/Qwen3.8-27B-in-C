#define _POSIX_C_SOURCE 200809L

#include "tokenizer.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "Qwen3.8-27B-IQ4_XS.gguf";
    printf("Testing Tokenizer on %s ...\n", model_path);

    gguf_context_t gguf;
    if (gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "Failed to open GGUF model %s\n", model_path);
        return 1;
    }

    tokenizer_t tok;
    if (tokenizer_init(&tok, &gguf) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        gguf_close(&gguf);
        return 1;
    }

    printf("Tokenizer initialized: vocab_size = %u, n_merges = %u\n", tok.vocab_size, tok.n_merges);

    const char *tests[] = {
        "Bonjour le monde !",
        "Hello, how are you today?",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "2 + 2 = 4",
        "Qwen 3.8-27B Hybrid GDN + Attention"
    };
    size_t n_tests = sizeof tests / sizeof tests[0];

    for (size_t i = 0; i < n_tests; i++) {
        uint32_t tokens[256];
        uint32_t n = tokenizer_encode(&tok, tests[i], tokens, 256, false);
        printf("\nPrompt [%zu]: \"%s\"\nEncoded (%u tokens): [", i, tests[i], n);
        for (uint32_t j = 0; j < n; j++) {
            printf("%u%s", tokens[j], (j + 1 < n) ? ", " : "");
        }
        printf("]\nDecoded: \"");
        for (uint32_t j = 0; j < n; j++) {
            printf("%s", tokenizer_decode(&tok, tokens[j]));
        }
        printf("\"\n");
    }

    tokenizer_free(&tok);
    gguf_close(&gguf);
    printf("\n=== Tokenizer test completed successfully ===\n");
    return 0;
}
