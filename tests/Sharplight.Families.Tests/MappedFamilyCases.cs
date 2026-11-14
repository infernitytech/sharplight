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
    private static readonly int[] Jais2Ids = [102, 955, 428, 102, 955, 1009, 899, 1009, 38, 102, 633, 428, 831, 535, 75, 38, 955, 1009, 899, 1009, 393, 393, 899, 102];
    private static readonly int[] MellumIds = [284, 372, 193, 882, 837, 650, 835, 351, 351, 194, 713, 284, 713, 284, 936, 571, 936, 533, 533, 701, 533, 434, 777, 602];
    private static readonly int[] DbrxIds = [127, 66, 518, 974, 553, 697, 900, 55, 121, 457, 1001, 456, 637, 454, 413, 646, 60, 651, 884, 519, 44, 766, 848, 2];
    private static readonly int[] Mistral4Ids = [505, 539, 72, 499, 752, 892, 325, 610, 338, 241, 662, 957, 502, 902, 979, 752, 892, 443, 713, 752, 892, 433, 43, 471];
    private static readonly int[] GraniteMoeSwaIds = [571, 412, 744, 569, 985, 115, 180, 319, 399, 186, 551, 455, 264, 553, 952, 760, 172, 635, 585, 53, 799, 91, 819, 245];

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
        // Jais 2: layer norms, squared-ReLU MLP with biases.
        new("tiny-random-jais2 HF (jais2, transformers)", "hf/tiny-random-jais2", Prompt, "ids", Jais2Ids, ""),
        new("tiny-random-jais2 f32 GGUF (jais2, llama.cpp)", "gguf/tiny-random-jais2-f32.gguf", Prompt, "ids", Jais2Ids, ""),
        // Granite SWA: muP scales, sinks, sliding layers, a NoPE layer and a second rotary base (layer_rope_theta, which llama.cpp's converter does not
        // carry over, so the GGUF case uses the MoE checkpoint with one theta).
        new("tiny-random-granite-swa HF (granite_swa, transformers)", "hf/tiny-random-granite-swa", Prompt, "ids",
            [279, 531, 430, 882, 826, 179, 383, 108, 831, 753, 113, 961, 355, 615, 355, 974, 214, 686, 686, 686, 686, 262, 859, 859], ""),
        new("tiny-random-granitemoe-swa HF (granitemoe_swa, shared expert, transformers)", "hf/tiny-random-granitemoe-swa", Prompt, "ids", GraniteMoeSwaIds, ""),
        new("tiny-random-granitemoe-swa f32 GGUF (granite_swa, llama.cpp)", "gguf/tiny-random-granitemoe-swa-f32.gguf", Prompt, "ids", GraniteMoeSwaIds, ""),
        // Mellum 2: Qwen3-MoE layers with sliding layers on their own rope base and a dense layer (mlp_layer_types). The GGUF is checked against
        // transformers: llama.cpp b11177 rotates the sliding layers with the global base despite rope.freq_base_swa (it matches transformers once
        // both bases are equal).
        new("tiny-random-mellum HF (mellum, transformers)", "hf/tiny-random-mellum", Prompt, "ids",
            [246, 167, 157, 666, 163, 44, 265, 867, 271, 850, 234, 534, 465, 534, 813, 241, 866, 696, 371, 696, 805, 850, 716, 368], ""),
        new("tiny-random-mellum-llamacpp HF (mellum all-MoE, transformers)", "hf/tiny-random-mellum-llamacpp", Prompt, "ids", MellumIds, ""),
        new("tiny-random-mellum-llamacpp f32 GGUF (mellum, llama.cpp)", "gguf/tiny-random-mellum-llamacpp-f32.gguf", Prompt, "ids", MellumIds, ""),
        // DBRX: layer norms without bias, fused q/k/v clamped to +-clip_qkv, fused [E*F, H] experts.
        new("tiny-random-dbrx HF (dbrx, transformers)", "hf/tiny-random-dbrx", Prompt, "ids", DbrxIds, ""),
        new("tiny-random-dbrx f32 GGUF (dbrx, llama.cpp)", "gguf/tiny-random-dbrx-f32.gguf", Prompt, "ids", DbrxIds, ""),
        // Mistral 4: latent attention with YaRN and query temperature (beyond 16 positions), softmax routing over expert groups, stacked experts.
        new("tiny-random-mistral4 HF (mistral4, transformers)", "hf/tiny-random-mistral4", Prompt, "ids", Mistral4Ids, ""),
        // llama.cpp's converter (Mistral3ForConditionalGeneration path) drops the query temperature, so that GGUF follows llama.cpp, not transformers,
        // past position 16; with attention.temperature_scale / temperature_length added to the file llama.cpp and transformers agree.
        new("tiny-random-mistral4 f32 GGUF (mistral4, llama.cpp)", "gguf/tiny-random-mistral4-f32.gguf", Prompt, "ids",
            [505, 539, 72, 499, 448, 785, 593, 138, 878, 128, 560, 807, 128, 683, 287, 274, 96, 902, 923, 527, 304, 127, 878, 362], ""),
        new("tiny-random-mistral4-temperature f32 GGUF (mistral4 query temperature, llama.cpp)", "gguf/tiny-random-mistral4-temperature-f32.gguf", Prompt, "ids",
            Mistral4Ids, ""),
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
        // 23 tokens: llama.cpp's 24th is a near tie (log-prob margin 0.001).
        new("tiny-random-deepseek f32 GGUF (deepseek V1 dense lead, shared expert, unnormalized scaled top-k, llama.cpp)", "gguf/tiny-random-deepseek-f32.gguf", Prompt, "ids",
            [593, 248, 303, 176, 746, 405, 657, 36, 658, 237, 613, 780, 765, 958, 427, 467, 96, 383, 720, 246, 784, 804, 625], ""),
        new("tiny-random-bailingmoe f32 GGUF (bailingmoe head size from rope dims, shared expert, llama.cpp)", "gguf/tiny-random-bailingmoe-f32.gguf", Prompt, "ids",
            [300, 213, 142, 313, 242, 3, 23, 909, 549, 667, 667, 159, 734, 7, 125, 313, 832, 210, 714, 314, 36, 213, 563, 7], ""),
    ];
}
