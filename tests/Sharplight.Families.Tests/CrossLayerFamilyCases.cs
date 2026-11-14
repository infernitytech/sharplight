using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Families of bring-up round 9 (docs/model-support.md) whose layers connect across depth: hierarchical layer cycles (HRM-Text), ... Tiny random
/// Hugging Face checkpoints come from <c>tools/mk_tiny_decoders.py</c>, their references are greedy ids and prompt logits from transformers
/// (<c>tools/hf_greedy_reference.py --logits</c>); GGUF files are converted from them with llama.cpp's converter (<c>tools/convert_tiny_gguf.py</c>)
/// and checked against llama.cpp 8ca5a80. Every case skips when its files are missing.
/// </summary>
internal static class CrossLayerFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] HrmTextIds = [286, 749, 614, 927, 904, 919, 119, 57, 606, 981, 331, 254, 583, 883, 787, 953, 59, 125, 98, 828, 425, 136, 184, 1006];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // HRM-Text (tools/mk_tiny_decoders.py hrm-text): a low and a high stack of 2 layers in 2 high cycles of 2 low passes and a high pass (12 KV
        // slots), weight-less norms, per-channel sigmoid attention gates, the learned low-state init. llama.cpp 8ca5a80 gives the same ids (CPU, Vulkan).
        new("tiny-random-hrm-text HF (hrm_text, transformers)", "hf/tiny-random-hrm-text", Prompt, "ids", HrmTextIds, ""),
        new("tiny-random-hrm-text f32 GGUF (hrm_text, llama.cpp)", "gguf/tiny-random-hrm-text-f32.gguf", Prompt, "ids", HrmTextIds, ""),
    ];
}
