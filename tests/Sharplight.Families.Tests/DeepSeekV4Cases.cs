using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// DeepSeek-V4: compressed-KV attention (a sliding window of 8 keys plus compressed entries of 4 positions with an indexer selecting 4 of them,
/// and of 128 positions attended densely), hyper-connections with Sinkhorn mixing, hash-routed first layers and clamped SwiGLU experts. The tiny
/// random checkpoint comes from <c>tools/mk_tiny_deepseek_v4.py</c>; its references are the greedy ids and prompt logits of transformers 5.20
/// (<c>tools/hf_greedy_reference.py --logits</c>). The prompt is long enough (160 tokens) for the layer with 128-position entries to attend one.
/// The indexer's top-k picks of the random model have near-ties (relative score gaps down to 1e-3) that f16 cache rounding flips, so the cases
/// keep the caches in f32. The GGUF (<c>tools/convert_tiny_gguf.py</c>) gives the same 24 ids in llama.cpp 8ca5a80 on the CPU with f32 caches and
/// flash attention off (<c>-ctk f32 -ctv f32 -fa off -fit off</c>); with its defaults it flips a pick at the tenth token. The Vulkan runs keep the
/// compressed-cache ops on the CPU fallback. Every case skips when its files are missing.
/// </summary>
internal static class DeepSeekV4Cases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art, music, food and fashion. "
        + "Visitors came from every corner of the world to walk along the river, to see the old churches and the new iron tower, and to sit in the cafes "
        + "where painters, writers and students talked late into the night about politics, love and the future of the century that had just begun. "
        + "Many of them stayed for years.";

    private static readonly int[] V4Ids = [402, 524, 707, 295, 998, 998, 998, 458, 276, 573, 829, 519, 158, 26, 903, 518, 461, 775, 607, 405, 980, 892, 465, 329];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        new("tiny-random-deepseek-v4 HF (deepseek_v4, transformers)", "hf/tiny-random-deepseek-v4", Prompt, "ids", V4Ids, "", NeedsCpuFallback: true, F32KvCache: true),
        new("tiny-random-deepseek-v4 f32 GGUF (deepseek4, llama.cpp)", "gguf/tiny-random-deepseek-v4-f32.gguf", Prompt, "ids", V4Ids, "", NeedsCpuFallback: true, F32KvCache: true),
    ];
}
