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
    private static readonly int[] Cohere2MoeIds = [747, 3, 30, 630, 225, 225, 225, 656, 685, 411, 7, 747, 29, 98, 929, 372, 859, 11, 319, 470, 747, 29, 567, 11];
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
        // Hunyuan v4: latent attention with output gate and sinks, indexer in its dense regime, clamped SwiGLU experts, hyper-connections (3 streams).
        new("tiny-random-hy-v4 HF (hy_v4, transformers)", "hf/tiny-random-hy-v4", Prompt, "ids",
            [954, 350, 912, 711, 295, 223, 710, 662, 175, 970, 611, 547, 737, 462, 661, 35, 414, 81, 604, 808, 906, 627, 420, 522], ""),
        // The same model with DeepSeek-V4's manifold-constrained hyper-connections (Sinkhorn stream mixing, tools/mk_tiny_hy_v4_mhc.py).
        new("tiny-random-hy-v4-mhc HF (hy_v4 + DeepSeek-V4 mHC, transformers)", "hf/tiny-random-hy-v4-mhc", Prompt, "ids",
            [954, 513, 60, 245, 773, 761, 713, 734, 7, 830, 644, 813, 624, 367, 165, 189, 509, 806, 825, 99, 784, 652, 986, 21], ""),
        // llama.cpp's converter (Mistral3ForConditionalGeneration path) drops the query temperature, so that GGUF follows llama.cpp, not transformers,
        // past position 16; with attention.temperature_scale / temperature_length added to the file llama.cpp and transformers agree.
        new("tiny-random-mistral4 f32 GGUF (mistral4, llama.cpp)", "gguf/tiny-random-mistral4-f32.gguf", Prompt, "ids",
            [505, 539, 72, 499, 448, 785, 593, 138, 878, 128, 560, 807, 128, 683, 287, 274, 96, 902, 923, 527, 304, 127, 878, 362], ""),
        new("tiny-random-mistral4-temperature f32 GGUF (mistral4 query temperature, llama.cpp)", "gguf/tiny-random-mistral4-temperature-f32.gguf", Prompt, "ids",
            Mistral4Ids, ""),
        // Cohere2-MoE: parallel layer-norm blocks, a rotary dense first layer, sliding rotary MoE layers and a global NoPE layer, a shared expert averaged
        // with the routed sum.
        new("tiny-random-cohere2-moe HF (cohere2_moe, transformers)", "hf/tiny-random-cohere2-moe", Prompt, "ids", Cohere2MoeIds, ""),
        new("tiny-random-cohere2-moe f32 GGUF (cohere2moe, llama.cpp)", "gguf/tiny-random-cohere2-moe-f32.gguf", Prompt, "ids", Cohere2MoeIds, ""),
        // llama.cpp-only families: tiny random GGUF files from tools/mk_tiny_gguf.py (llama.cpp tensor names and keys), greedy ids of llama.cpp b11177 (CPU).
        new("tiny-random-refact f32 GGUF (refact ALiBi, llama.cpp)", "gguf/tiny-random-refact-f32.gguf", Prompt, "ids",
            [673, 918, 695, 714, 45, 621, 140, 303, 638, 159, 173, 52, 228, 500, 159, 217, 403, 329, 934, 450, 13, 44, 157, 627], ""),
        new("tiny-random-jais f32 GGUF (jais ALiBi, llama.cpp)", "gguf/tiny-random-jais-f32.gguf", Prompt, "ids",
            [15, 756, 745, 845, 745, 845, 745, 281, 692, 845, 845, 745, 281, 692, 845, 845, 745, 281, 692, 219, 845, 15, 845, 15], ""),
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
            [4, 915, 261, 657, 362, 921, 107, 576, 822, 573, 200, 646, 881, 1015, 607, 419, 999, 1018, 116, 22, 28, 1006, 875, 799], ""),
        // 23 tokens: llama.cpp's 24th is a near tie (log-prob margin 0.001).
        new("tiny-random-deepseek f32 GGUF (deepseek V1 dense lead, shared expert, unnormalized scaled top-k, llama.cpp)", "gguf/tiny-random-deepseek-f32.gguf", Prompt, "ids",
            [593, 248, 303, 176, 746, 405, 657, 36, 658, 237, 613, 780, 765, 958, 427, 467, 96, 383, 720, 246, 784, 804, 625], ""),
        new("tiny-random-bailingmoe f32 GGUF (bailingmoe head size from rope dims, shared expert, llama.cpp)", "gguf/tiny-random-bailingmoe-f32.gguf", Prompt, "ids",
            [300, 213, 142, 313, 242, 3, 23, 909, 549, 667, 667, 159, 734, 7, 125, 313, 832, 210, 714, 314, 36, 213, 563, 7], ""),
        // PLM: llama.cpp b11177 needs -fa off for it (its flash-attention path fails on value heads narrower than the key heads).
        new("tiny-random-plm f32 GGUF (plm latent attention, squared ReLU, llama.cpp)", "gguf/tiny-random-plm-f32.gguf", Prompt, "ids",
            [963, 201, 799, 975, 558, 169, 855, 545, 234, 639, 43, 975, 909, 201, 903, 855, 381, 331, 934, 826, 104, 652, 558, 903], ""),
        new("tiny-random-talkie f32 GGUF (talkie weight-less norms, per-head q gain after rope, embedding skip, logit scale, llama.cpp)", "gguf/tiny-random-talkie-f32.gguf", Prompt, "ids",
            [69, 593, 233, 498, 328, 862, 557, 872, 505, 302, 527, 842, 524, 90, 642, 25, 288, 324, 949, 603, 621, 595, 339, 419], ""),
        new("tiny-random-nanbeige f32 GGUF (nanbeige two layer loops with the output norm between them, llama.cpp)", "gguf/tiny-random-nanbeige-f32.gguf", Prompt, "ids",
            [944, 185, 39, 453, 183, 365, 572, 160, 339, 530, 834, 848, 834, 642, 715, 715, 230, 585, 344, 262, 715, 838, 502, 619], ""),
        new("tiny-random-arctic f32 GGUF (arctic dense MLP plus a separately normed expert branch, llama.cpp)", "gguf/tiny-random-arctic-f32.gguf", Prompt, "ids",
            [106, 184, 220, 695, 488, 192, 1022, 140, 112, 904, 524, 4, 368, 735, 1005, 15, 184, 97, 781, 225, 1009, 626, 195, 488], ""),
        new("tiny-random-deci f32 GGUF (deci per-layer heads and widths, linear-attention and attention-free layers, llama.cpp)", "gguf/tiny-random-deci-f32.gguf", Prompt, "ids",
            [828, 449, 523, 730, 172, 528, 15, 810, 229, 642, 855, 1017, 953, 452, 402, 901, 503, 573, 108, 495, 1014, 108, 379, 1017], ""),
        new("tiny-random-grok f32 GGUF (grok scaled soft-capped attention, sandwich norms, GELU experts plus dense MLP, logit cap, llama.cpp)", "gguf/tiny-random-grok-f32.gguf", Prompt, "ids",
            [546, 813, 108, 853, 841, 470, 224, 546, 800, 241, 118, 118, 546, 285, 747, 632, 632, 377, 853, 800, 904, 800, 904, 964], ""),
        // Maple, Kolibri-1 and Ricursive: a newer llama.cpp (b11177 lacks the latter two and clamps Maple's experts the Step-3.5 way); its text
        // output, matched byte for byte by these ids.
        new("tiny-random-maple f32 GGUF (maple NoPE global layers, sliding window, clamped SwiGLU experts, llama.cpp)", "gguf/tiny-random-maple-f32.gguf", Prompt, "ids",
            [883, 210, 498, 883, 498, 94, 579, 498, 498, 498, 878, 498, 878, 498, 878, 498, 878, 498, 216, 498, 216, 498, 216, 498], ""),
        // LFM3 is newer than b11177: reference ids from llama.cpp 8ca5a80 (CPU, greedy over libllama).
        new("tiny-random-lfm3 f32 GGUF (lfm3 short convs, sliding layers on their own base, partial rope by frequency factors, head gate, sandwich norms, llama.cpp)",
            "gguf/tiny-random-lfm3-f32.gguf", Prompt, "ids",
            [285, 285, 1008, 437, 460, 781, 775, 279, 151, 874, 968, 677, 888, 294, 388, 659, 120, 307, 225, 937, 602, 464, 783, 874], ""),
        new("tiny-random-maion-coder f32 GGUF (maion-coder NoPE full layers, sandwich norms, experts in a latent space, llama.cpp 8ca5a80)",
            "gguf/tiny-random-maion-coder-f32.gguf", Prompt, "ids",
            [625, 315, 49, 387, 716, 49, 49, 592, 233, 638, 130, 730, 135, 930], ""),   // cut before step 14, where llama.cpp's top two differ by 3e-4
        new("tiny-random-beam f32 GGUF (beam norm-preserving residual, power temperature on NoPE layers, biased head gate, sigmoid experts, softcap, llama.cpp 8ca5a80)",
            "gguf/tiny-random-beam-f32.gguf", Prompt, "ids",
            [98, 914, 609, 904, 75, 768, 564, 364, 698, 436, 939, 861, 967, 389, 520, 36, 578, 726, 620, 98, 903, 655, 27, 243], ""),
        new("tiny-random-muse-spark f32 GGUF (muse-spark learned residual weights, head output norm, latent experts with in/out norms, llama.cpp 8ca5a80)",
            "gguf/tiny-random-muse-spark-f32.gguf", Prompt, "ids",
            [432, 958, 117, 785, 827, 79, 707, 1010, 232, 232, 403, 301, 1000, 347, 644, 598, 216, 232, 597, 149, 492, 653, 37, 591], ""),
        // Clover: the tiny Qwen3.5-MoE checkpoint converted by llama.cpp and retargeted (tools/retarget_gguf.py: arch clover_moe, no M-RoPE sections,
        // no NextN block, one shared expert); hidden size 8, so the logits are flat (llama.cpp's top two differ by 0.0006 to 0.005 at steps 5, 7 and 15).
        new("tiny-random-clover-moe f32 GGUF (clover_moe gated delta net + gated attention, MoE with gated shared expert, llama.cpp 8ca5a80)",
            "gguf/tiny-random-clover-moe-f32.gguf", Prompt, "ids", [132029, 22595, 212119, 133773, 4415, 198543, 219592, 20547, 220921, 172932, 151129, 212119, 133773, 239941, 239941, 219592, 127405, 224729, 212119, 133773, 239941, 118789, 69541, 215913], ""),
        // Encoder-decoder (tools/mk_tiny_t5.py; references from hf_greedy_reference.py, whose "prompt logits" are the first decoder step): the original T5
        // (ReLU, output scaled by d_model^-0.5, bias saturating at distance 12, more decoder than encoder layers) and T5 v1.1 (gated tanh-GELU,
        // attention width 48 next to d_model 64); both cut before a close call (top two within 0.014 of 3, 0.22 of 30) that f16 GPU paths may flip.
        new("tiny-random-t5 HF (t5 relu, scaled tied head, relative position bias, cross-attention, transformers)", "hf/tiny-random-t5", Prompt, "ids",
            [1014, 9202, 19881, 19881, 19881, 19881, 19881, 30559, 30559, 30559, 6135, 22296, 19306, 7712, 7712, 19716], ""),
        new("tiny-random-t5-v1_1 HF (t5 gated-gelu, transformers)", "hf/tiny-random-t5-v1_1", Prompt, "ids",
            [8847, 26621, 8889, 8889, 31376, 731, 731, 28428, 28176, 8298, 27172, 9762, 7827, 7827, 7827, 29378, 28529], ""),
        new("flan-t5-small HF (t5 gated-gelu, separate LM head, transformers)", "hf/flan-t5-small", "Write a short story about a cat who learns to fly.", "ids",
            [37, 1712, 19, 7070, 5, 216, 19, 3, 9, 3, 89, 3822, 5, 216, 19, 3, 9, 3, 89, 3822, 5, 216, 19, 3], ""),
        // BART family on the same adapter (tools/mk_tiny_bart.py): post-norm LayerNorms, learned positions (offset 2), embedding norm, final logits bias;
        // mBART: pre-norm with final norms and scaled embeddings.
        new("tiny-random-bart HF (bart post-norm, learned positions, transformers)", "hf/tiny-random-bart", Prompt, "ids",
            [847, 634, 955, 634, 955, 634, 955, 634, 634, 955, 634, 634, 955, 634], ""),
        new("tiny-random-mbart HF (mbart pre-norm, scaled embeddings, transformers)", "hf/tiny-random-mbart", Prompt, "ids",
            [862, 862, 862, 862, 862, 862, 862, 862, 862, 862, 862, 862, 862, 862, 862, 862], ""),
        new("bart-base HF (bart, real checkpoint, transformers)", "hf/bart-base", Prompt, "ids",
            [0, 133, 812, 9, 1470, 16, 10, 343, 14, 6, 11, 5, 76, 23137, 8, 1684, 6, 21, 684, 13, 171, 383, 215, 25], ""),
        // Encoder-decoder: the prompt is encoded, the decoder starts from the pad token (llama.cpp b11177 llama-completion prints the same text).
        new("flan-t5-small Q4_K_M GGUF (t5 encoder-decoder, relative position bias, cross-attention, llama.cpp)", "gguf/flan-t5-small/flan-t5-small.Q4_K_M.gguf",
            "Write a short story about a cat who learns to fly.", "ids", [37, 1712, 19, 7070, 16, 8, 799, 5, 216, 19, 3, 9, 3, 89, 3822, 5, 216, 19, 3, 9, 3, 89, 3822, 5], ""),
        new("LFM3-1.7B Q8_0 GGUF (lfm3, llama.cpp 8ca5a80)", "gguf/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf", "The capital of France is", "ids",
            [4741, 63908, 597, 5205, 302, 3888, 355, 18943, 63908, 597, 5205, 302, 5881, 355, 2076, 59405, 63908, 597, 5205, 302, 5177, 355, 36038, 63908], ""),
        new("tiny-random-step35 f32 GGUF (step35 per-layer heads, half rope on full layers, head gate, sigmoid experts, clamped SwiGLU, llama.cpp)", "gguf/tiny-random-step35-f32.gguf", Prompt, "ids",
            [817, 475, 244, 560, 259, 346, 19, 862, 229, 587, 978, 519, 142, 725, 987, 180, 765, 891, 519, 162, 138, 168, 903, 41], ""),
        new("tiny-random-kolibri1 f32 GGUF (kolibri1 logit selection bias, sigmoid experts, NoPE global layers, sandwich norms, llama.cpp)", "gguf/tiny-random-kolibri1-f32.gguf", Prompt, "ids",
            [431, 327, 402, 402, 77, 77, 25, 402, 773, 773, 767, 767, 960, 887, 888, 674, 455, 999, 333, 582, 938, 71, 164, 527], ""),
        new("tiny-random-ricursive f32 GGUF (ricursive value residual, channel gate, NoPE global layer, clamped SwiGLU, llama.cpp)", "gguf/tiny-random-ricursive-f32.gguf", Prompt, "ids",
            [169, 627, 119, 886, 874, 601, 729, 631, 100, 407, 503, 590, 1004, 961, 503, 797, 224, 719, 850, 503, 193, 193, 700, 231], ""),
        // Transformers references (ids and last-prompt-position logits) of tiny checkpoints from mk_tiny_decoders.py, cut before a close call:
        // PhiMoE's 19th step and Kolibri-1's 18th differ by less than 0.01 between the top two logits.
        new("tiny-random-phimoe HF (phimoe layer norms, sparsemixer routing, sliding window, transformers)", "hf/tiny-random-phimoe", Prompt, "ids",
            [948, 961, 961, 539, 539, 107, 665, 150, 948, 217, 254, 973, 884, 1016, 100, 401, 358, 546], ""),
        // The same checkpoint converted by llama.cpp's converter (architecture renamed to PhiMoEForCausalLM, original_max_position_embeddings added, GPT-2
        // vocabulary; see tools/convert_phimoe_gguf.py). Checked against transformers: llama.cpp runs phimoe with RMS norms and softmax routing.
        new("tiny-random-phimoe f32 GGUF (phimoe layer norms, sparsemixer routing, transformers)", "gguf/tiny-random-phimoe-f32.gguf", Prompt, "ids",
            [948, 961, 961, 539, 539, 107, 665, 150, 948, 217, 254, 973, 884, 1016, 100, 401, 358, 546], ""),
        new("tiny-random-kolibri1 HF (kolibri1 logit selection bias, sandwich norms, NoPE global layer, transformers)", "hf/tiny-random-kolibri1", Prompt, "ids",
            [298, 196, 886, 442, 914, 489, 442, 661, 583, 292, 73, 997, 209, 829, 1003, 684, 935], ""),
        // Real checkpoints: bigcode/tiny_starcoder_py (gpt_bigcode, 164M) against transformers; inceptionai/jais-family-590m converted by llama.cpp's
        // converter (f16, muP scales folded into the embedding and the output head) against llama.cpp b11177.
        new("tiny_starcoder_py HF (gpt_bigcode, transformers)", "hf/tiny_starcoder_py", "def fibonacci(n):", "ids",
            [284, 415, 310, 610, 225, 34, 44, 291, 442, 225, 34, 284, 813, 44, 291, 442, 28176, 34682, 26, 96, 31, 35, 27, 474], ""),
        new("jais-family-590m f16 GGUF (jais ALiBi, llama.cpp)", "gguf/jais-family-590m/jais-family-590m-f16.gguf", "The capital of France is", "ids",
            [330, 5199, 371, 14707, 14, 1248, 455, 330, 10308, 371, 330, 15309, 371, 330, 1824, 725, 13, 417, 13, 18913, 932, 14, 1248, 455], ""),
        // Nanbeige4.2-3B: 22 layers looped twice (44 KV slots) with the output norm between the loops; llama.cpp b11177 text up to the first
        // close call (after " Pluto." the top two differ by 0.2 nats; on "The capital of France is" two llama.cpp runs with different thread counts already part at token 10).
        new("Nanbeige4.2-3B Q4_K_M GGUF (nanbeige layer loops, llama.cpp)", "gguf/Nanbeige4.2-3B/Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf",
            "The planets of the solar system in order from the Sun are Mercury, Venus, Earth, Mars,", "text", [], " Jupiter, Saturn, Uranus, Neptune, and Pluto.", Tokens: 11),
    ];
}
