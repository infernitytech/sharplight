// Greedy reference ids (and the logits of every step) from llama.cpp, for tiny GGUF files of architectures that have no transformers class
// (hy_v4_flash). Prints a JSON object: the prompt ids, the generated ids, the smallest top-1/top-2 logit margin and each step's logits.
//
//   cc -O2 -o llamacpp_greedy llamacpp_greedy.c -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<llama.cpp>/build/bin -lllama -lggml -lggml-base
//   LD_LIBRARY_PATH=<llama.cpp>/build/bin ./llamacpp_greedy <model.gguf> <prompt> <n_new> [logits: 0/1]
#include "llama.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void quiet(enum ggml_log_level level, const char * text, void * data) {
    (void) level; (void) text; (void) data;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_new> [logits]\n", argv[0]);
        return 1;
    }
    const int n_new = atoi(argv[3]);
    const int dump = argc > 4 && atoi(argv[4]) != 0;
    llama_log_set(quiet, NULL);
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token prompt[4096];
    const int n_prompt = llama_tokenize(vocab, argv[2], (int) strlen(argv[2]), prompt, 4096, false, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    printf("{\"prompt\": [");
    for (int i = 0; i < n_prompt; ++i) {
        printf(i ? ", %d" : "%d", prompt[i]);
    }
    printf("], ");

    struct llama_batch batch = llama_batch_get_one(prompt, n_prompt);
    llama_token ids[4096];
    float margin = FLT_MAX;
    char * logits_json = NULL;
    size_t logits_len = 0;
    FILE * logits_out = open_memstream(&logits_json, &logits_len);
    for (int step = 0; step < n_new; ++step) {
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step);
            return 1;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        float second = -FLT_MAX;
        for (int v = 0; v < n_vocab; ++v) {
            if (v != best && logits[v] > second) {
                second = logits[v];
            }
        }
        if (logits[best] - second < margin) {
            margin = logits[best] - second;
        }
        if (dump) {
            fprintf(logits_out, step ? ", [" : "[");
            for (int v = 0; v < n_vocab; ++v) {
                fprintf(logits_out, v ? ", %.6g" : "%.6g", logits[v]);
            }
            fprintf(logits_out, "]");
        }
        ids[step] = best;
        batch = llama_batch_get_one(&ids[step], 1);
    }
    fclose(logits_out);

    printf("\"ids\": [");
    for (int i = 0; i < n_new; ++i) {
        printf(i ? ", %d" : "%d", ids[i]);
    }
    printf("], \"margin\": %.6g", margin);
    if (dump) {
        printf(", \"logits\": [%s]", logits_json);
    }
    printf("}\n");
    free(logits_json);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
