using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Inkling (Thinking Machines, bring-up round 11 of docs/model-support.md). The tiny checkpoint and its reference come from
/// <c>tools/mk_tiny_inkling.py</c> (transformers' InklingForCausalLM), the GGUF from the unsloth fork's converter; every case skips when its
/// files are missing.
/// </summary>
internal static class InklingFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] InklingIds = [280, 418, 444, 878, 468, 564, 875, 107, 437, 214, 864, 19, 376, 858, 780, 947, 712, 33, 368, 712, 511, 515, 442, 373];

    private static readonly int[] Inkling2Ids = [210, 206, 393, 715, 832, 234, 565, 427, 490, 481, 796, 274, 35, 553, 306, 126, 934, 749, 562, 987, 735, 36, 164, 201];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Sliding-window layers with one kv head and full layers with two, residual short convolutions of k / v / attention / MLP outputs, the
        // query-conditioned relative bias, the log-floor temperature (floor 8), router-scored shared experts, embedding norm, logits divided by
        // the width multiplier and cut to the unpadded vocabulary (published model.llm.* layout with interleaved w13 projections).
        new("tiny-random-inkling HF (inkling_mm_model)", "hf/tiny-random-inkling", Prompt, "ids", InklingIds, ""),
        // The same checkpoint through the fork's converter (tools/convert_tiny_gguf.py, GPT-2 pre-tokenizer); the fork's llama.cpp gives the
        // reference ids.
        new("tiny-random-inkling f32 GGUF (inkling, unsloth llama.cpp)", "gguf/tiny-random-inkling-f32.gguf", Prompt, "ids", InklingIds, ""),
        // Inkling 2 Mini (inkling2_loop, tools/mk_tiny_inkling2.py): block attention residuals over normalized rows, per-token sinks, output gates,
        // L2 clips, static and per-token key / value convolutions, softcapped SwiGLU experts, the loop over layers 2-5 run 3 times with message
        // gates, the final logit soft cap and the unpadded vocabulary, in the release's layout (transformers 5.24 reference).
        new("tiny-random-inkling2 HF (inkling2_loop)", "hf/tiny-random-inkling2", Prompt, "ids", Inkling2Ids, ""),
    ];
}
