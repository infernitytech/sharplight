using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Linear-attention hybrids: Kimi-Linear and Ling 3 (delta attention with a decay per key channel next to latent attention, on the latent decoder) and
/// MiniMax-01 (lightning attention, a decay-only linear attention with fixed per-head slopes, next to softmax attention). The tiny float32
/// checkpoints come from <c>tools/mk_tiny_decoders.py</c> (transformers 5.20); references are greedy ids and prompt logits from transformers
/// (<c>tools/hf_greedy_reference.py --logits</c>). Every case skips when its files are missing.
/// </summary>
internal static class LinearAttentionFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] KimiLinearIds = [557, 11, 454, 503, 751, 672, 454, 385, 944, 952, 13, 911, 677, 155, 45, 827, 460, 405, 842, 897, 370, 530, 232, 708];
    private static readonly int[] BailingHybridIds = [993, 61, 888, 389, 1011, 903, 269, 297, 320, 898, 247, 417, 258, 60, 97, 170, 348, 847, 381, 89, 437, 49, 180, 880];
    private static readonly int[] MiniMaxIds = [830, 765, 140, 69, 325, 1019, 707, 784, 359, 498, 815, 198, 45, 784, 770, 149, 450, 325, 607, 1006, 476, 253, 815, 435];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Kimi-Linear: three delta-attention layers (low-rank decay and output-gate projections) around a latent-attention layer without rope,
        // sigmoid routing over two groups with a correction bias and a shared expert, a dense first layer.
        new("tiny-random-kimi-linear-f32 HF (kimi_linear, transformers)", "hf/tiny-random-kimi-linear-f32", Prompt, "ids", KimiLinearIds, ""),
        // Ling 3: delta attention with the bounded gate, latent attention with rope and a per-head output gate every third layer, two dense layers.
        new("tiny-random-bailing-hybrid HF (bailing_hybrid, transformers)", "hf/tiny-random-bailing-hybrid", Prompt, "ids", BailingHybridIds, ""),
        // MiniMax-01: lightning attention every other layer, residuals from the normalized input with (alpha, beta) scales, softmax-routed experts.
        new("tiny-random-minimax HF (minimax, transformers)", "hf/tiny-random-minimax", Prompt, "ids", MiniMaxIds, ""),
    ];
}
