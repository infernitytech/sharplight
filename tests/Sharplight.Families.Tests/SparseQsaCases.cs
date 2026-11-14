using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Qwen3.8-Flash-Next (qwen4_exp) with its QSA indexer in the sparse regime (bring-up round 9): the tiny checkpoint of
/// <c>tools/mk_tiny_qwen4exp_sparse.py</c> keeps two blocks of 4 plus the incomplete tail, so every attention row past the 11th token attends
/// only part of its context. The ids and the prompt logits (<c>reference_logits.json</c>) are those of transformers
/// (<c>tools/hf_greedy_reference.py</c>); llama.cpp 8ca5a80 gives the same ids from the GGUF of <c>tools/convert_tiny_gguf.py</c> on the CPU and
/// on Vulkan, with and without flash attention. Every case skips when its files are missing.
/// </summary>
internal static class SparseQsaCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] Ids =
        [582, 179235, 135671, 236167, 201405, 171483, 125883, 225464, 111463, 198197, 194166, 233043, 209845, 211806, 63541, 23281, 183150, 157701, 220545, 24156, 23178, 87851, 12478, 25273];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        new("tiny-random-qwen3.8-flash-next-sparse HF (QSA block selection, transformers)", "hf/tiny-random-qwen3.8-flash-next-sparse", Prompt, "ids", Ids, ""),
        new("tiny-random-qwen3.8-flash-next-sparse f32 GGUF (qwen4exp, llama.cpp)", "gguf/tiny-random-qwen3.8-flash-next-sparse-f32.gguf", Prompt, "ids", Ids, ""),
    ];
}
