using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Language models of the vision-language families of bring-up round 9 (docs/model-support.md): HunyuanOCR / Hunyuan-VL and CogVLM. Tiny random Hugging
/// Face checkpoints come from the generators in <c>tools/</c>, their references are greedy ids and prompt logits from transformers
/// (<c>tools/hf_greedy_reference.py --logits</c>); GGUF files are converted from them with llama.cpp's converter and checked against llama.cpp
/// 8ca5a80. The image paths are checked in Sharplight.Vision.Tests. Every case skips when its files are missing.
/// </summary>
internal static class VisionLanguageFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] CogVlmIds = [15780, 25540, 18773, 15184, 15587, 12650, 9139, 2746, 29063, 29478, 6142, 8029, 31028, 12650, 28734, 8258, 24699, 21290, 5234, 12650, 28734, 24119, 2735, 21290];

    private static readonly int[] HunyuanVlIds = [59622, 24861, 5795, 91930, 37659, 67261, 107420, 85384, 78941, 81115, 18770, 43229, 42868, 68484, 72004, 39478, 83952, 8156, 35454, 93929, 76593, 16008];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // HunyuanOCR (tools/mk_tiny_hunyuan_vl.py): the Hunyuan dense layer with XD-RoPE sections [4, 4, 4, 4] over the elements of a 32-wide head;
        // text tokens carry their position on all four axes. The last two reference steps are near ties (logit margins 0.49), so the cases stop
        // before them. llama.cpp 8ca5a80 continues identically (CPU and Vulkan).
        new("tiny-random-hunyuan-vl HF (hunyuan_vl, transformers)", "hf/tiny-random-hunyuan-vl", Prompt, "ids", HunyuanVlIds, ""),
        new("tiny-random-hunyuan-vl f32 GGUF (hunyuan_vl, llama.cpp)", "gguf/tiny-random-hunyuan-vl-f32.gguf", Prompt, "ids", HunyuanVlIds, ""),
        // CogVLM (tools/mk_tiny_cogvlm.py, references from the checkpoint's remote code): Llama layers with visual experts; text rows use the
        // language experts only. Smallest reference logit margin 0.082. llama.cpp 8ca5a80 computes the same rows (its per-row prompt logits match
        // the reference), but its cogvlm graph never selects the output rows (no build_inp_out_ids), so after a multi-token prompt it samples the
        // first row's logits (the BOS position): its first token is 1997, and its continuation from there matches the reference forward exactly
        // (CPU and Vulkan).
        new("tiny-random-cogvlm HF (cogvlm, remote code)", "hf/tiny-random-cogvlm", Prompt, "ids", CogVlmIds, ""),
        new("tiny-random-cogvlm f32 GGUF (cogvlm, llama.cpp)", "gguf/tiny-random-cogvlm-f32.gguf", Prompt, "ids", CogVlmIds, ""),
    ];
}
