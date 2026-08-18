/* Debug tokenizer : pourquoi "Ġ"+"2" ne fusionnent-ils pas ? */
#include "model.h"
#include "tokenizer.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    qwen_model_t m;
    if (model_init(&m, argv[1], 64) != 0) return 1;
    tokenizer_t tok;
    if (tokenizer_init(&tok, &m.gguf) != 0) return 1;

    /* token 220 : quel texte exactement ? */
    printf("vocab[220] = \"%s\" (hex: ", tok.vocab[220].text);
    for (const unsigned char *p = (const unsigned char *)tok.vocab[220].text; *p; p++)
        printf("%02X ", *p);
    printf(")\n");
    printf("vocab[17]  = \"%s\"\n", tok.vocab[17].text);
    printf("byte_to_str[' '] = hex: ");
    for (const unsigned char *p = (const unsigned char *)tok.byte_to_str[' ']; *p; p++)
        printf("%02X ", *p);
    printf("\n\n");

    /* cherche les tokens simples */
    uint32_t id_G = 0, id_2 = 0;
    for (uint32_t i = 0; i < tok.vocab_size; i++) {
        if (strcmp(tok.vocab[i].text, "Ġ") == 0) id_G = i;
        if (strcmp(tok.vocab[i].text, "2") == 0 && i < 300) id_2 = i;
    }
    printf("token \"Ġ\" = %u   token \"2\" (bas) = %u\n", id_G, id_2);
    printf("existe \"Ġ2\" ? ");
    uint32_t id_G2 = 0;
    for (uint32_t i = 0; i < tok.vocab_size; i++)
        if (strcmp(tok.vocab[i].text, "Ġ2") == 0) { id_G2 = i; break; }
    printf("%u\n", id_G2);

    /* combien de merges ont leurs deux parties résolues ? */
    uint32_t ok = 0, ko = 0;
    for (uint32_t i = 0; i < tok.n_merges; i++) {
        if (tok.merges[i].left_id != UINT32_MAX && tok.merges[i].right_id != UINT32_MAX) ok++;
        else ko++;
    }
    printf("merges résolus: %u / %u (échoués: %u)\n", ok, tok.n_merges, ko);

    /* les 5 premiers merges */
    for (uint32_t i = 0; i < 5 && i < tok.n_merges; i++)
        printf("merge[%u] = (%u, %u)\n", i, tok.merges[i].left_id, tok.merges[i].right_id);

    /* tokens Ġ+chiffre existants ? */
    printf("tokens Ġ+chiffre : ");
    int found = 0;
    for (uint32_t i = 0; i < tok.vocab_size && found < 8; i++) {
        const char *t = tok.vocab[i].text;
        if (t[0] == (char)0xC4 && t[1] == (char)0xA0 && t[2] >= '0' && t[2] <= '9') {
            printf("id=%u \"%s\"  ", i, t);
            found++;
        }
    }
    if (!found) printf("AUCUN");
    printf("\n");

    /* tokens spéciaux résolus ? */
    printf("bos=%u eos=%u unk=%u pad=%u\n", tok.bos_id, tok.eos_id, tok.unk_id, tok.pad_id);

    /* merge (he) : "hello" doit faire 1 token */
    uint32_t tbuf[32];
    uint32_t n = tokenizer_encode(&tok, "hello", tbuf, 32, false);
    printf("\"hello\" -> [");
    for (uint32_t i = 0; i < n; i++) printf("%u ", tbuf[i]);
    printf("]\n");

    /* round-trip accents */
    n = tokenizer_encode(&tok, "café déjà", tbuf, 32, false);
    printf("\"café déjà\" -> [");
    for (uint32_t i = 0; i < n; i++) printf("%u ", tbuf[i]);
    printf("]\n");
    printf("  decode round-trip : ");
    for (uint32_t i = 0; i < n; i++) printf("%s", tokenizer_decode(&tok, tbuf[i]));
    printf("\n");

    /* token 220 décodé = espace ? */
    printf("decode(220) = [%s]\n", tokenizer_decode(&tok, 220));

    tokenizer_free(&tok);
    model_free(&m);
    return 0;
}
