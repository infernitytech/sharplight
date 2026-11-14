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
        // llama.cpp-only families: tiny random GGUF files from tools/mk_tiny_gguf.py (llama.cpp tensor names and keys), greedy ids of llama.cpp b11177 (CPU).
        // ALiBi (Refact, Jais, Baichuan-13B) runs on the CPU provider when a GPU is used.
        new("tiny-random-refact f32 GGUF (refact ALiBi, llama.cpp)", "gguf/tiny-random-refact-f32.gguf", Prompt, "ids",
            [673, 918, 695, 714, 45, 621, 140, 303, 638, 159, 173, 52, 228, 500, 159, 217, 403, 329, 934, 450, 13, 44, 157, 627], "", NeedsCpuFallback: true),
        new("tiny-random-jais f32 GGUF (jais ALiBi, llama.cpp)", "gguf/tiny-random-jais-f32.gguf", Prompt, "ids",
            [15, 756, 745, 845, 745, 845, 745, 281, 692, 845, 845, 745, 281, 692, 845, 845, 745, 281, 692, 219, 845, 15, 845, 15], "", NeedsCpuFallback: true),
        new("tiny-random-orion f32 GGUF (orion, llama.cpp)", "gguf/tiny-random-orion-f32.gguf", Prompt, "ids",
            [104, 837, 472, 487, 153, 487, 472, 861, 652, 572, 476, 95, 472, 861, 215, 76, 81, 568, 564, 664, 104, 104, 104, 447], ""),
        new("tiny-random-codeshell f32 GGUF (codeshell, embedding through the output head, llama.cpp)", "gguf/tiny-random-codeshell-f32.gguf", Prompt, "ids",
            [100, 100, 100, 100, 785, 869, 925, 869, 869, 869, 869, 869, 869, 869, 869, 869, 869, 869, 464, 869, 464, 869, 869, 869], ""),
        new("tiny-random-plamo f32 GGUF (plamo parallel residual, llama.cpp)", "gguf/tiny-random-plamo-f32.gguf", Prompt, "ids",
            [174, 720, 355, 510, 513, 781, 13, 673, 1017, 735, 809, 745, 400, 849, 829, 499, 744, 859, 359, 694, 653, 809, 745, 411], ""),
        new("tiny-random-plamo3 f32 GGUF (plamo3 sliding window, llama.cpp)", "gguf/tiny-random-plamo3-f32.gguf", Prompt, "ids",
            [894, 449, 562, 448, 60, 740, 44, 21, 577, 577, 98, 410, 789, 849, 849, 962, 47, 666, 299, 569, 611, 598, 311, 293], ""),
        new("tiny-random-pangu-embedded f32 GGUF (pangu-embedded, llama.cpp)", "gguf/tiny-random-pangu-embedded-f32.gguf", Prompt, "ids",
            [113, 619, 445, 958, 840, 792, 661, 621, 281, 258, 935, 853, 900, 666, 709, 169, 215, 257, 47, 833, 93, 793, 668, 322], ""),
        new("tiny-random-maincoder f32 GGUF (maincoder q/k norm after rope, llama.cpp)", "gguf/tiny-random-maincoder-f32.gguf", Prompt, "ids",
            [767, 104, 767, 883, 833, 493, 654, 475, 12, 377, 108, 733, 956, 895, 272, 615, 883, 477, 157, 41, 1020, 733, 938, 36], ""),
        new("tiny-random-spark2_5 f32 GGUF (spark2_5 per-head gate, sliding window, llama.cpp)", "gguf/tiny-random-spark2_5-f32.gguf", Prompt, "ids",
            [566, 656, 830, 819, 185, 542, 646, 784, 583, 669, 781, 598, 432, 222, 859, 927, 569, 894, 220, 475, 529, 335, 125, 366], ""),
        new("tiny-random-baichuan f32 GGUF (baichuan 32 layers rotary, llama.cpp)", "gguf/tiny-random-baichuan-f32.gguf", Prompt, "ids",
            [1004, 786, 523, 263, 550, 862, 713, 331, 802, 17, 567, 668, 567, 806, 213, 725, 862, 564, 932, 626, 892, 959, 978, 942], ""),
        new("tiny-random-baichuan-alibi f32 GGUF (baichuan 40 layers ALiBi, llama.cpp)", "gguf/tiny-random-baichuan-alibi-f32.gguf", Prompt, "ids",
            [4, 915, 261, 657, 362, 921, 107, 576, 822, 573, 200, 646, 881, 1015, 607, 419, 999, 1018, 116, 22, 28, 1006, 875, 799], "", NeedsCpuFallback: true),
    ];
}
