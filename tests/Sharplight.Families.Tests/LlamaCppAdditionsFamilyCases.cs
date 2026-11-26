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

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Domyn Edge (domynedge, remote code): sandwich norms, unweighted q norm and weighted k norm, rope only on the sliding-window layers, whose
        // heads are half the size of the full layers' (per-layer head sizes and attention scales). Smallest margin 0.21 (seed 1, weight std 0.08).
        new("tiny-random-domynedge HF (domynedge, remote code)", "hf/tiny-random-domynedge", Prompt, "ids", DomynEdgeIds, ""),
        new("tiny-random-domynedge f32 GGUF (domynedge, llama.cpp)", "gguf/tiny-random-domynedge-hf-f32.gguf", Prompt, "ids", DomynEdgeIds, ""),
    ];
}
