// lldump.c: llama.cpp logits dumper used in benchmarks section 87 (build: gcc -O2 lldump.c -I$LLAMA_SRC/include -I$LLAMA_SRC/ggml/include -L$RUNTIME -lllama -lggml-base -lggml).
// lldump <model.gguf> <ids comma list> <out.f32> <ngl> [ncmoe] : prompt processed in one batch, last-row logits written as float32;
// then (optional) tokens fed one by one: env LLDUMP_SERIAL=1 decodes the prompt token by token instead.
#include "llama.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage\n"); return 2; }
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[4]);
    // LLDUMP_CPU_TENSORS=<regex>: those tensors stay in host memory (as llama.cpp's -ot <regex>=CPU / -ncmoe).
    static struct llama_model_tensor_buft_override overrides[2];
    if (getenv("LLDUMP_CPU_TENSORS")) {
        overrides[0].pattern = getenv("LLDUMP_CPU_TENSORS");
        overrides[0].buft = ggml_backend_cpu_buffer_type();
        overrides[1].pattern = NULL;
        overrides[1].buft = NULL;
        mp.tensor_buft_overrides = overrides;
    }
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    int ids[4096]; int n = 0;
    for (char * s = strtok(argv[2], ","); s; s = strtok(NULL, ",")) ids[n++] = atoi(s);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = 8; cp.n_threads_batch = 8;
    if (getenv("LLDUMP_FA")) cp.flash_attn_type = atoi(getenv("LLDUMP_FA")) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (getenv("LLDUMP_SERIAL")) {
        for (int i = 0; i < n; i++) { if (llama_decode(ctx, llama_batch_get_one(&ids[i], 1))) { fprintf(stderr, "decode failed\n"); return 1; } }
    } else {
        if (llama_decode(ctx, llama_batch_get_one(ids, n))) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    const float * lg = llama_get_logits_ith(ctx, -1);
    int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen(argv[3], "wb"); fwrite(lg, sizeof(float), nv, f); fclose(f);
    int best = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    printf("n_vocab %d best %d %.4f\n", nv, best, lg[best]);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
