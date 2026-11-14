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
    ];
}
