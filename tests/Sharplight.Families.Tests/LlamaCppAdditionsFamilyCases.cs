using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Architectures llama.cpp added after the pinned parity commit 8ca5a80 (docs/model-support.md section 26). The tiny checkpoints come from
/// <c>tools/mk_tiny_remote_code.py</c>; their references are greedy ids from transformers running the shipped modeling code
/// (<c>tools/hf_greedy_reference.py</c>), and the GGUF files llama.cpp's converter writes from them give the same text on llama.cpp b12798
/// (9fa7a33) unless a case says otherwise. Every case skips when its files are missing.
/// </summary>
internal static class LlamaCppAdditionsFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    // Reference ids shared by a Hugging Face case and the GGUF converted from it. Declared before All, which reads them.
    private static readonly int[] DomynEdgeIds = [61641, 61641, 61641, 61641, 61641, 61641, 26466, 44786, 17650, 7904, 8735, 47416, 47416, 47416, 23678, 101963,
        101963, 64974, 103280, 15925, 105556, 9700, 9700, 9700];

    // Hunyuan v4 Flash (tools/mk_tiny_hy_v4_flash.py, no transformers class): greedy ids of llama.cpp HEAD a07831a (tools/llamacpp_greedy.c) on the
    // f32 GGUF its converter writes from the checkpoint, up to the tiny tokenizer's end of text (id 0, the 20th); smallest top-2 margin 0.062. 8 indexer heads: with fewer, pools often score exactly 0
    // (every head's dot negative), and llama.cpp's top-k breaks such ties in no defined order (the smallest pick margin is now 1.4%).
    private static readonly int[] HyV4FlashIds = [793, 48, 335, 991, 340, 772, 162, 899, 51, 988, 131, 773, 523, 262, 177, 831, 513, 756, 899];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Domyn Edge (domynedge, remote code): sandwich norms, unweighted q norm and weighted k norm, rope only on the sliding-window layers, whose
        // heads are half the size of the full layers' (per-layer head sizes and attention scales). Smallest margin 0.21 (seed 1, weight std 0.08).
        new("tiny-random-domynedge HF (domynedge, remote code)", "hf/tiny-random-domynedge", Prompt, "ids", DomynEdgeIds, ""),
        new("tiny-random-domynedge f32 GGUF (domynedge, llama.cpp)", "gguf/tiny-random-domynedge-hf-f32.gguf", Prompt, "ids", DomynEdgeIds, ""),

        // Hunyuan v4 Flash (hy_v4_flash): sparse layers with the pooled indexer (pools of 4, index_topk 8: the best earlier pool plus the query's
        // own) and sliding-window layers (window 8, twice the heads), sandwich norms, latent MoE, hyper-connections and two Engram modules.
        // F32 KV cache: the pooled keys come from f16 cache rows otherwise, which can swap the indexer's picks.
        // The checkpoint itself (no transformers class: its reference_logits.json holds llama.cpp's prompt logits of the GGUF).
        new("tiny-random-hy-v4-flash HF (hy_v4_flash, llama.cpp reference)", "hf/tiny-random-hy-v4-flash", Prompt, "ids", HyV4FlashIds, "", F32KvCache: true),
        new("tiny-random-hy-v4-flash f32 GGUF (hy_v4_flash, llama.cpp)", "gguf/tiny-random-hy-v4-flash-f32.gguf", Prompt, "ids", HyV4FlashIds, "", F32KvCache: true),
    ];
}
