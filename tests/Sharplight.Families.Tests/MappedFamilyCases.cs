using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Families brought up as importer mappings onto the existing adapters (the "partial" architectures of docs/model-support.md). The tiny random Hugging Face
/// checkpoints come from <c>tools/mk_tiny_decoders.py</c>; their references are greedy ids and prompt logits from transformers
/// (<c>tools/hf_greedy_reference.py --logits</c>, which writes <c>reference_logits.json</c> next to the checkpoint). Tiny GGUF files are checked against llama.cpp b11177.
/// Every case skips when its files are missing.
/// </summary>
internal static class MappedFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    // Reference ids shared by a Hugging Face case and the GGUF converted from it (transformers and llama.cpp agree). Declared before All, which reads them.
    private static readonly int[] BigCodeIds = [66, 85, 139, 469, 469, 469, 469, 469, 799, 469, 469, 469, 469, 799, 901, 469, 469, 469, 469, 469, 469, 799, 194, 937];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Qwen2-MoE: softmax routing without renormalisation, sigmoid-gated shared expert, a dense layer between MoE layers.
        new("tiny-random-qwen2-moe HF (qwen2_moe, transformers)", "hf/tiny-random-qwen2-moe", Prompt, "ids",
            [563, 919, 1022, 411, 411, 411, 298, 567, 434, 411, 1013, 949, 74, 490, 74, 490, 444, 711, 812, 74, 613, 711, 115, 382], ""),
        new("tiny-random-qwen2-moe f32 GGUF (qwen2moe, llama.cpp)", "gguf/tiny-random-qwen2-moe-f32.gguf", Prompt, "ids",
            [991, 789, 313, 445, 453, 571, 445, 878, 647, 991, 991, 991, 789, 445, 58, 878, 114, 878, 313, 991, 991, 991, 991, 991], ""),
        // StarCoder 1 / SantaCoder (gpt_bigcode / starcoder): GPT-2 layout, linear weights, one key/value head.
        new("tiny-random-gpt-bigcode HF (gpt_bigcode, transformers)", "hf/tiny-random-gpt-bigcode", Prompt, "ids", BigCodeIds, ""),
        new("tiny-random-gpt-bigcode f32 GGUF (starcoder, llama.cpp)", "gguf/tiny-random-gpt-bigcode-f32.gguf", Prompt, "ids", BigCodeIds, ""),
    ];
}
