using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// MoE-variant families of bring-up round 9 (docs/model-support.md). The tiny checkpoints and their references come from the generators in
/// <c>tools/</c>; every case skips when its files are missing.
/// </summary>
internal static class MoeVariantFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] GroveMoeIds = [960, 13, 781, 885, 393, 1009, 427, 961, 1009, 54, 3, 95, 669, 110, 515, 764, 669, 885, 819, 781, 143, 669, 493, 2];

    private static readonly int[] K2HorizonIds = [792, 314, 345, 345, 345, 345, 879, 166, 147, 761, 748, 521, 839, 40, 363, 715, 105, 11, 166, 147, 761, 583, 820, 240];

    private static readonly int[] Xing40Ids = [15599, 25286, 5575, 23023, 17173, 31556, 25014, 11472, 29810, 14478, 6023, 22590, 17316, 4804, 18125, 19242, 31802, 3260, 1589, 22086, 7806, 4200, 20213, 1879];

    private static readonly int[] Xing40UnclampedIds = [6112, 668, 11268, 30978, 24563, 13488, 17008, 24779, 1860, 17173, 23014, 25646, 14778, 13284, 12232, 22140, 22482, 20945, 18322, 11235, 8539, 19486, 17220, 10931];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // GroveMoE (tools/mk_tiny_grovemoe.py, no transformers class): Qwen3-MoE with top-4 of 8 experts plus one adjugate expert per pair, against
        // transformers' Qwen3-MoE with the remote code's MoE block. llama.cpp 8ca5a80 continues differently from the 14th token: its graph feeds
        // the adjugate experts the routed experts' output and weights them by the probability of expert id / 2 (the generator's --llamacpp mode
        // reproduces its ids), so the GGUF is checked against the reference ids too.
        new("tiny-random-grovemoe HF (GroveMoeForCausalLM, remote-code MoE block)", "hf/tiny-random-grovemoe", Prompt, "ids", GroveMoeIds, ""),
        new("tiny-random-grovemoe f32 GGUF (grovemoe, reference ids)", "gguf/tiny-random-grovemoe-f32.gguf", Prompt, "ids", GroveMoeIds, ""),
        // K2-Horizon (tools/mk_tiny_k2_horizon.py with the model's remote code): a dense layer, then MoE layers whose values come from routed value
        // experts (MoVA), sigmoid routing with selection biases and a shared expert, RMS norms over 2 groups, per-head q/k norms, the softplus
        // output gate and YaRN. llama.cpp 8ca5a80 gives the same ids.
        new("tiny-random-k2-horizon HF (k2_horizon, remote code)", "hf/tiny-random-k2-horizon", Prompt, "ids", K2HorizonIds, ""),
        new("tiny-random-k2-horizon f32 GGUF (k2-horizon, llama.cpp)", "gguf/tiny-random-k2-horizon-f32.gguf", Prompt, "ids", K2HorizonIds, ""),
        // Xing4.0 (tools/mk_tiny_xing4_0.py with the model's remote code): DeepSeek-V3's latent attention and sigmoid-routed MoE layers on 4
        // hyper-connection streams with Xing's mixing (logits clamped to ±30, no eps on the collapse or after the softmax, row-wise mixing)
        // averaged before the output norm. The first model's first layer drives its mixing logits beyond the clamp, which llama.cpp 8ca5a80's
        // fused mixing kernel leaves out (it continues differently from the 4th token), so its GGUF is checked against the reference ids; the
        // unclamped model stays inside the clamp, and llama.cpp gives its reference ids.
        new("tiny-random-xing4-0 HF (xing4_0, remote code, clamped mixing)", "hf/tiny-random-xing4-0", Prompt, "ids", Xing40Ids, ""),
        new("tiny-random-xing4-0 f32 GGUF (xing4_0, reference ids)", "gguf/tiny-random-xing4-0-f32.gguf", Prompt, "ids", Xing40Ids, ""),
        new("tiny-random-xing4-0-unclamped HF (xing4_0, remote code)", "hf/tiny-random-xing4-0-unclamped", Prompt, "ids", Xing40UnclampedIds, ""),
        new("tiny-random-xing4-0-unclamped f32 GGUF (xing4_0, llama.cpp)", "gguf/tiny-random-xing4-0-unclamped-f32.gguf", Prompt, "ids", Xing40UnclampedIds, ""),
    ];
}
