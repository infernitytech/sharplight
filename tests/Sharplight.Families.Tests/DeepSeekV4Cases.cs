using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// DeepSeek-V4: compressed-KV attention (a sliding window of 8 keys plus compressed entries of 4 positions with an indexer selecting 4 of them,
/// and of 128 positions attended densely), hyper-connections with Sinkhorn mixing, hash-routed first layers and clamped SwiGLU experts. The tiny
/// random checkpoint comes from <c>tools/mk_tiny_deepseek_v4.py</c>; its references are the greedy ids and prompt logits of transformers 5.20
/// (<c>tools/hf_greedy_reference.py --logits</c>). The prompt is long enough (160 tokens) for the layer with 128-position entries to attend one.
/// The indexer's top-k picks of the random model have near-ties (relative score gaps down to 1e-3) that f16 cache rounding flips, so the cases
/// keep the caches in f32. The GGUF (<c>tools/convert_tiny_gguf.py</c>) gives the same 24 ids in llama.cpp 8ca5a80 on the CPU with f32 caches and
/// flash attention off (<c>-ctk f32 -ctv f32 -fa off -fit off</c>); with its defaults it flips a pick at the tenth token. The Vulkan runs keep every
/// op on the GPU. Every case skips when its files are missing.
/// DeepSeek-V4.1 (no transformers class): the tiny GGUF of <c>tools/mk_tiny_deepseek_v41.py</c> (entries of 1 and 2 positions, layers attending
/// an earlier layer's compressed caches and selection, candidate blocks, index keys derived from the entries, Engram n-gram memory, single-pass
/// hyper-connections); its reference ids come from the float64 numpy port of llama.cpp's graph, <c>tools/ref_deepseek_v41.py</c> (smallest
/// indexer gap 1.9e-4), and llama.cpp gives the same 24 ids when it decodes the prompt one token at a time (f32 caches, flash attention off). Its
/// one-batch prompt decode differs from the second token on; on variants of the model this needs a layer with entries of one position.
/// </summary>
internal static class DeepSeekV4Cases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art, music, food and fashion. "
        + "Visitors came from every corner of the world to walk along the river, to see the old churches and the new iron tower, and to sit in the cafes "
        + "where painters, writers and students talked late into the night about politics, love and the future of the century that had just begun. "
        + "Many of them stayed for years.";

    private static readonly int[] V41Ids = [664, 916, 915, 789, 353, 815, 759, 492, 670, 511, 587, 282, 126, 403, 819, 410, 447, 525, 750, 819, 245, 253, 835, 564];

    private static readonly int[] V4Ids = [402, 524, 707, 295, 998, 998, 998, 458, 276, 573, 829, 519, 158, 26, 903, 518, 461, 775, 607, 405, 980, 892, 465, 329];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        new("tiny-random-deepseek-v4 HF (deepseek_v4, transformers)", "hf/tiny-random-deepseek-v4", Prompt, "ids", V4Ids, "", F32KvCache: true),
        new("tiny-random-deepseek-v4 f32 GGUF (deepseek4, llama.cpp)", "gguf/tiny-random-deepseek-v4-f32.gguf", Prompt, "ids", V4Ids, "", F32KvCache: true),
        new("tiny-random-deepseek-v41 GGUF (deepseek41, numpy reference)", "gguf/tiny-random-deepseek-v41.gguf", Prompt, "ids", V41Ids, "", F32KvCache: true),
    ];
}
